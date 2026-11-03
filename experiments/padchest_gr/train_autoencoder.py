"""
Train grounded autoencoder on ViT embeddings.

Encoder: Linear(768, F=78, bias=False)  — projects embedding to concept space
Decoder: W_E^T                           — tied weights (fix_D mode)

Loss:
    L = lam * MSE(h_hat, h) + (1 - lam) * BCE(c, z_gt)

BCE masks unknowns (z_gt = 0) and applies per-concept pos_weight.

Faithfulness (monitoring only): pass h_hat through the frozen ViT classifier
head and compare predictions with y_hat (ViT's predictions on original h).

Data split: deterministic random 80/20 on the autoencoder dataset.

Usage:
    uv run experiments/padchest_gr/train_autoencoder.py lam=0.5 run_name=lam05
    uv run experiments/padchest_gr/train_autoencoder.py lam=0.0 bias=true
"""

import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from pathlib import Path
from sklearn.metrics import roc_auc_score

from experiments.padchest.train_mlp_classifier import MLPClassifier
from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

logger = None  # set in main() via setup_logger

ROOT = Path(get_root_dir())
DATA_DIR = Path(os.getenv("DATA_DIR"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR"))

EMBEDDINGS_PATH = ROOT / "data" / "padchest_final" / "embeddings_vit.pt"


class GroundedAutoencoder(nn.Module):
    """Encoder projects h -> c; decoder is tied (W_D = W_E^T) reconstructing h."""

    def __init__(self, d: int, n_concepts: int, bias: bool = False):
        super().__init__()
        self.encoder = nn.Linear(d, n_concepts, bias=bias)

    def forward(self, h):
        c = self.encoder(h)                 # [B, F]
        h_hat = F.linear(c, self.encoder.weight.t())  # [B, d]
        return h_hat, c


def compute_loss(h, h_hat, c, z_gt, lam, pos_weight=None):
    """Loss = lam * recon + (1-lam) * cls, with unknown masking."""
    recon = F.mse_loss(h_hat, h)

    known = (z_gt != 0)                     # [B, F]
    if known.any():
        target = ((z_gt + 1) / 2).float()   # -1 -> 0, +1 -> 1
        bce = F.binary_cross_entropy_with_logits(
            c, target, pos_weight=pos_weight, reduction="none",
        )
        cls = bce[known].mean()
    else:
        cls = torch.tensor(0.0, device=h.device)

    total = lam * recon + (1 - lam) * cls
    return total, recon, cls


def _load_classifier(device, ckpt_path: str | None = None):
    """Load the frozen ViT classifier head for faithfulness evaluation.

    If ckpt_path is None, falls back to the latest ViT run under
    checkpoints/padchest/ via latest.txt. Returns a Linear(d -> K) in eval mode.
    """
    if ckpt_path is None:
        pad_ckpt_dir = ROOT / "checkpoints" / "padchest"
        vit_runs = sorted(d for d in os.listdir(pad_ckpt_dir) if d.startswith("train_cls_vit_b_16_"))
        latest_run = pad_ckpt_dir / vit_runs[-1]
        with open(latest_run / "latest.txt") as f:
            ckpt_path = f.read().strip()

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    w = state["classifier.weight"]
    b = state["classifier.bias"]

    head = nn.Linear(w.shape[1], w.shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded ViT classifier head: {w.shape[1]} -> {w.shape[0]}  ({ckpt_path})")
    return head, None  # no extra diagnosis_names override


def _load_mlp_classifier(ckpt_path: Path, device):
    """Load the MLP head trained on frozen ConceptCLIP embeddings.

    Returns the MLP (in eval mode, requires_grad disabled) and the
    diagnosis_names list stored in the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MLPClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dims=list(ckpt["hidden_dims"]),
        output_dim=ckpt["output_dim"],
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        f"Loaded MLP classifier: "
        f"{ckpt['input_dim']} -> {'->'.join(map(str, ckpt['hidden_dims']))} "
        f"-> {ckpt['output_dim']}  (val_macro_auc={ckpt.get('val_macro_auc', float('nan')):.4f})  "
        f"({ckpt_path})"
    )
    return model, list(ckpt["diagnosis_names"])


def load_classifier(args, device):
    """Dispatch on args.classifier_type. Returns (callable, diagnosis_names|None)."""
    ctype = args.get("classifier_type", "mlp")
    if ctype == "vit":
        ckpt_path = args.get("classifier_ckpt", None)
        return _load_classifier(device, ckpt_path=ckpt_path)
    if ctype == "mlp":
        ckpt_path = args.get("classifier_ckpt", None)
        if ckpt_path is None:
            raise ValueError("classifier_type=mlp requires classifier_ckpt in config")
        return _load_mlp_classifier(ROOT / ckpt_path, device)
    raise ValueError(f"Unknown classifier_type: {ctype!r} (expected 'mlp' or 'vit')")


@torch.no_grad()
def evaluate(model, h, z_gt, y_hat, classifier, concept_names):
    """Compute MSE, per-concept AUC, and faithfulness metrics.

    y_hat is the frozen classifier's hard predictions on the original h.
    Faithfulness measures agreement between classifier(h_hat) and y_hat.
    """
    model.eval()
    h_hat, c = model(h)

    mse = F.mse_loss(h_hat, h).item()

    # Per-concept AUC (only on known labels)
    c_probs = torch.sigmoid(c).cpu().numpy()
    z_np = z_gt.cpu().numpy()
    aucs = {}
    for j, name in enumerate(concept_names):
        known = z_np[:, j] != 0
        if known.sum() < 2:
            aucs[name] = None
            continue
        labels = ((z_np[known, j] + 1) / 2).astype(int)
        scores = c_probs[known, j]
        if len(set(labels)) < 2:
            aucs[name] = None
        else:
            aucs[name] = roc_auc_score(labels, scores)

    # Faithfulness: classifier(h_hat) vs. classifier(h) hard predictions.
    logits_recon = classifier(h_hat)                    # [N, K]
    probs_recon = torch.sigmoid(logits_recon)
    preds_recon = (probs_recon > 0.5).float()
    y_hat_sl = y_hat                                  # [N, K]

    # Balanced accuracy: mean over valid classes of (TPR + TNR) / 2.
    # Valid = class has both positives and negatives in this batch.
    # Constant-class predictor = 0.5, perfect = 1.0.
    tp = ((preds_recon == 1) & (y_hat_sl == 1)).float().sum(dim=0)
    tn = ((preds_recon == 0) & (y_hat_sl == 0)).float().sum(dim=0)
    pos = y_hat_sl.sum(dim=0)
    neg = (1 - y_hat_sl).sum(dim=0)
    tpr = tp / pos.clamp(min=1)
    tnr = tn / neg.clamp(min=1)
    bal = 0.5 * (tpr + tnr)
    valid = (pos > 0) & (neg > 0)
    agreement = float(bal[valid].mean().item()) if valid.any() else float("nan")

    ce = F.binary_cross_entropy(probs_recon, y_hat_sl).item()

    return mse, aucs, agreement, ce


@hydra.main(
    version_base=None,
    config_path=os.path.join(str(ROOT), "configs"),
    config_name="autoencoder",
)
def main(args: DictConfig):
    global logger
    # Use hydra's output dir for the log file (so print -> file AND terminal)
    import hydra as _h
    hydra_cfg = _h.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg.runtime.output_dir
    run_name = args.run_name or f"ae_lam{args.lam}"
    logger = setup_logger(log_dir=log_dir, filename=f"{run_name}.log")

    # Load embeddings
    embeddings_path = ROOT / "data" / args.get("dataset_dir", "padchest_final") / args.embeddings_file
    logger.info(f"Loading {embeddings_path} ...")
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]       # [N, d]
    z_gt = data["z_gt"]                   # [N, F]
    concept_names = data["concept_names"]  # [F]
    diagnosis_names = data.get("diagnosis_names")
    N, d = embeddings.shape
    F_dim = z_gt.shape[1]
    logger.info(f"  N={N}, d={d}, F={F_dim}")
    if diagnosis_names is not None:
        logger.info(f"  Diagnoses in embeddings file: {list(diagnosis_names)}")

    # Load the pre-computed iterative-stratified split (from make_split.py)
    split_path = ROOT / "data" / args.get("dataset_dir", "padchest_final") / "split.pt"
    if split_path.exists():
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        logger.info(f"  Loaded split from {split_path}: train={len(train_idx)}, val={len(val_idx)}")
    else:
        logger.info(f"  No {split_path} — falling back to random split (seed={args.seed})")
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(N, generator=g)
        n_train = int(N * args.train_ratio)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    import random as _random
    _random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    logger.info(f"Seed: {args.seed}")

    h_train = embeddings[train_idx].to(device)
    h_val = embeddings[val_idx].to(device)
    z_train = z_gt[train_idx].to(device)
    z_val = z_gt[val_idx].to(device)

    # pos_weight per concept
    n_pos = (z_train == 1).float().sum(dim=0)
    n_neg = (z_train == -1).float().sum(dim=0)
    pos_weight = (n_neg / n_pos.clamp(min=1)).to(device)
    logger.info(f"  pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")

    # Model
    model = GroundedAutoencoder(d, F_dim, bias=args.bias).to(device)
    with torch.no_grad():
        w = torch.randn_like(model.encoder.weight)
        w = w / w.norm(dim=1, keepdim=True)
        model.encoder.weight.copy_(w)

    # Faithfulness classifier: ViT head or MLP, selected by args.classifier_type.
    classifier, ckpt_diag_names = load_classifier(args, device)
    if ckpt_diag_names is not None:
        diagnosis_names = ckpt_diag_names
        logger.info(f"  Classifier diagnosis names: {diagnosis_names}")

    # Compute y_hat on val once from the frozen classifier (used for faithfulness).
    with torch.no_grad():
        logits_val = classifier(h_val)
        y_hat_val = (torch.sigmoid(logits_val) > 0.5).float()
    K = y_hat_val.shape[1]
    logger.info(f"  Faithfulness eval on K={K} classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # LR schedule: "cosine" (warmup + cosine decay to lr_min), "step", or null.
    schedule = args.get("lr_schedule", None)
    warmup_epochs = int(args.get("warmup_epochs", 0))
    lr_min = float(args.get("lr_min", 1e-7))
    if schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        cosine_epochs = max(args.epochs - warmup_epochs, 1)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr_min)
        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                              total_iters=warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup, cosine],
                                     milestones=[warmup_epochs])
        else:
            scheduler = cosine
        logger.info(f"  LR schedule: cosine (warmup={warmup_epochs}, lr_min={lr_min})")
    elif schedule == "step":
        step_size = int(args.get("step_size", 100))
        gamma = float(args.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma,
        )
        logger.info(f"  LR schedule: step (step_size={step_size}, gamma={gamma})")
    else:
        scheduler = None
        logger.info("  LR schedule: none (constant)")

    # Wandb
    from omegaconf import OmegaConf
    run_name = args.run_name or f"ae_lam{args.lam}"
    wandb.init(project=args.wandb_project, name=run_name,
               config=OmegaConf.to_container(args, resolve=True))

    # Epoch 0 eval
    mse, aucs, faith_agree, faith_ce = evaluate(
        model, h_val, z_val, y_hat_val, classifier, concept_names,
    )
    valid_aucs = [v for v in aucs.values() if v is not None]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    def _auc_summary(aucs_dict):
        """Return (mean, median, p25, p75, min, max) over valid AUCs. NaN if none."""
        vals = np.asarray([v for v in aucs_dict.values() if v is not None], dtype=float)
        if vals.size == 0:
            nan = float("nan")
            return nan, nan, nan, nan, nan, nan
        return (
            float(vals.mean()),
            float(np.median(vals)),
            float(np.quantile(vals, 0.25)),
            float(np.quantile(vals, 0.75)),
            float(vals.min()),
            float(vals.max()),
        )

    mean_auc, med_auc, p25_auc, p75_auc, min_auc, max_auc = _auc_summary(aucs)
    logger.info(f"\nEpoch 0 (before training)")
    logger.info(f"  MSE: {mse:.6f}  AUC mean/median: {mean_auc:.4f}/{med_auc:.4f}  "
                f"Faith agree: {faith_agree:.4f}  CE: {faith_ce:.4f}")
    wandb.log({
        "val/mse": mse,
        "val/auc_mean": mean_auc,
        "val/auc_median": med_auc,
        "val/auc_p25": p25_auc,
        "val/auc_p75": p75_auc,
        "val/auc_min": min_auc,
        "val/auc_max": max_auc,
        "val/faith_agree": faith_agree,
        "val/faith_ce": faith_ce,
        "epoch": 0,
    }, step=0)

    best_mean_auc = mean_auc
    ckpt_dir = ROOT / "checkpoints" / args.get("ckpt_subdir", "autoencoder_gr")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    samples_seen = 0
    N_train = h_train.shape[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N_train, device=device)

        for start in range(0, N_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            h_b = h_train[idx]
            z_b = z_train[idx]

            h_hat, c = model(h_b)
            loss, recon, cls = compute_loss(h_b, h_hat, c, z_b, args.lam, pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Unit-sphere renormalization
            if not args.no_normalize:
                with torch.no_grad():
                    w = model.encoder.weight
                    w.div_(w.norm(dim=1, keepdim=True))

            samples_seen += len(idx)
            wandb.log({
                "train/loss": loss.item(),
                "train/recon": recon.item(),
                "train/cls": cls.item(),
            }, step=samples_seen)

        # Eval every epoch
        mse, aucs, faith_agree, faith_ce = evaluate(
            model, h_val, z_val, y_hat_val, classifier, concept_names,
        )
        mean_auc, med_auc, p25_auc, p75_auc, min_auc, max_auc = _auc_summary(aucs)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:>3}  MSE: {mse:.6f}  "
                  f"AUC mean/median: {mean_auc:.4f}/{med_auc:.4f}  "
                  f"Faith agree: {faith_agree:.4f}  CE: {faith_ce:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "val/mse": mse,
            "val/auc_mean": mean_auc,
            "val/auc_median": med_auc,
            "val/auc_p25": p25_auc,
            "val/auc_p75": p75_auc,
            "val/auc_min": min_auc,
            "val/auc_max": max_auc,
            "val/faith_agree": faith_agree,
            "val/faith_ce": faith_ce,
            "train/lr": current_lr,
            "epoch": epoch,
        }, step=samples_seen)

        if scheduler is not None:
            scheduler.step()

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            ckpt_path = ckpt_dir / f"ae_{run_name}.pt"
            torch.save({
                "encoder_weight": model.encoder.weight.detach().cpu(),
                "encoder_bias": model.encoder.bias.detach().cpu() if args.bias else None,
                "epoch": epoch,
                "mean_auc": mean_auc,
                "d": d,
                "n_concepts": F_dim,
                "concept_names": concept_names,
                "args": OmegaConf.to_container(args, resolve=True),
            }, ckpt_path)
            logger.info(f"  ** New best: {mean_auc:.4f} saved to {ckpt_path}")

    wandb.finish()
    logger.info(f"\nDone. Best mean AUC: {best_mean_auc:.4f}")


if __name__ == "__main__":
    main()
