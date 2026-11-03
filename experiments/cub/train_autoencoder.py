"""
Train grounded autoencoder on frozen CLIP embeddings for CUB.

Encoder: Linear(D=768, F=312, bias=true)  — projects CLIP embedding to concept space
Decoder: W_E^T                              — tied weights

Loss:
    L = lam * MSE(h_hat, h) + (1 - lam) * BCE(c, z_gt)

BCE masks unknowns (z_gt = 0) and applies per-concept pos_weight.

Faithfulness: pass h_hat through the frozen MLP classifier head, compare
argmax to the classifier's argmax on h (single-label, 200 species).
Faith metric is top-1 agreement (single-label analog of balanced accuracy).

Train/val split: from `is_train` field in the embeddings file (the official
80/20 CUB split).

Usage:
    uv run experiments/cub/train_autoencoder.py
    uv run experiments/cub/train_autoencoder.py lam=0.1 run_name=ae_cub_lam0.1
"""

import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from experiments.cub.train_mlp_classifier import MLPClassifier
from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

logger = None
ROOT = Path(get_root_dir())


class GroundedAutoencoder(nn.Module):
    """Encoder projects h -> c; decoder is tied (W_D = W_E^T) reconstructing h."""

    def __init__(self, d: int, n_concepts: int, bias: bool = False):
        super().__init__()
        self.encoder = nn.Linear(d, n_concepts, bias=bias)

    def forward(self, h):
        c = self.encoder(h)                                  # [B, F]
        h_hat = F.linear(c, self.encoder.weight.t())          # [B, d]
        return h_hat, c


def compute_loss(h, h_hat, c, z_gt, lam, pos_weight=None):
    """Loss = lam * recon + (1-lam) * cls, with unknown masking."""
    recon = F.mse_loss(h_hat, h)
    known = (z_gt != 0)                                      # [B, F]
    if known.any():
        target = ((z_gt + 1) / 2).float()                    # -1 -> 0, +1 -> 1
        bce = F.binary_cross_entropy_with_logits(
            c, target, pos_weight=pos_weight, reduction="none",
        )
        cls = bce[known].mean()
    else:
        cls = torch.tensor(0.0, device=h.device)
    total = lam * recon + (1 - lam) * cls
    return total, recon, cls


def load_classifier(ckpt_path: Path, device):
    """Load the frozen MLP classifier head."""
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
        f"-> {ckpt['output_dim']}  (val_top1={ckpt.get('val_top1', float('nan')):.4f})  "
        f"({ckpt_path})"
    )
    return model


@torch.no_grad()
def evaluate(model, h, z_gt, y_hat, classifier, concept_names):
    """Compute MSE, per-concept AUC, faith_top1, faith_ce.

    Faithfulness for single-label: top-1 agreement between
    argmax(classifier(h_hat)) and argmax(classifier(h)).
    Cross-entropy: CE between classifier(h_hat) softmax and y_hat (argmax of orig).
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

    # Faithfulness: classifier(h_hat) vs y_hat (classifier argmax on h)
    logits_recon = classifier(h_hat)                          # [N, K]
    pred_recon = logits_recon.argmax(dim=-1)                  # [N]
    faith_top1 = (pred_recon == y_hat).float().mean().item()

    # Soft faith: CE between recon softmax and y_hat as one-hot target
    faith_ce = F.cross_entropy(logits_recon, y_hat).item()

    return mse, aucs, faith_top1, faith_ce


@hydra.main(
    version_base=None,
    config_path=os.path.join(str(ROOT), "configs"),
    config_name="autoencoder_cub",
)
def main(args: DictConfig):
    global logger
    import hydra as _h
    hydra_cfg = _h.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg.runtime.output_dir
    run_name = args.run_name or f"ae_cub_lam{args.lam}"
    logger = setup_logger(log_dir=log_dir, filename=f"{run_name}.log")

    # Load embeddings + concept labels + class labels
    embeddings_path = ROOT / "data" / args.get("dataset_dir", "CUB") / args.embeddings_file
    logger.info(f"Loading {embeddings_path} ...")
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]                           # [N, d]
    z_gt = data["z_gt"]                                       # [N, F]
    is_train = data["is_train"].bool()
    concept_names = data["concept_names"]                     # [F]
    class_names = data["class_names"]                         # [K]
    N, d = embeddings.shape
    F_dim = z_gt.shape[1]
    K = len(class_names)
    logger.info(f"  N={N}, d={d}, F={F_dim}, K={K}")
    logger.info(f"  train={int(is_train.sum())}, val={int((~is_train).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    import random as _random
    _random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    logger.info(f"Seed: {args.seed}")

    h_train = embeddings[is_train].to(device)
    h_val = embeddings[~is_train].to(device)
    z_train = z_gt[is_train].to(device)
    z_val = z_gt[~is_train].to(device)

    # pos_weight per concept (counts {-1, +1} known cells; {0} skipped)
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

    # Frozen classifier (MLP head trained on these CLIP embeddings)
    classifier = load_classifier(ROOT / args.classifier_ckpt, device)

    # Compute y_hat on val from the frozen classifier (for faithfulness)
    with torch.no_grad():
        y_hat_val = classifier(h_val).argmax(dim=-1)          # [N_val]
    logger.info(f"  Faithfulness eval on K={K} classes (single-label)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # LR schedule
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

    # Wandb
    wandb.init(project=args.wandb_project, name=run_name,
               config=OmegaConf.to_container(args, resolve=True))

    def _auc_summary(aucs_dict):
        vals = np.asarray([v for v in aucs_dict.values() if v is not None],
                          dtype=float)
        if vals.size == 0:
            nan = float("nan")
            return nan, nan, nan, nan, nan, nan
        return (
            float(vals.mean()), float(np.median(vals)),
            float(np.quantile(vals, 0.25)), float(np.quantile(vals, 0.75)),
            float(vals.min()), float(vals.max()),
        )

    # Epoch 0 eval
    mse, aucs, faith_top1, faith_ce = evaluate(
        model, h_val, z_val, y_hat_val, classifier, concept_names,
    )
    mean_auc, med_auc, p25_auc, p75_auc, min_auc, max_auc = _auc_summary(aucs)
    logger.info(f"\nEpoch 0 (before training)")
    logger.info(f"  MSE: {mse:.6f}  AUC mean/median: {mean_auc:.4f}/{med_auc:.4f}  "
                f"Faith top1: {faith_top1:.4f}  CE: {faith_ce:.4f}")
    wandb.log({
        "val/mse": mse, "val/auc_mean": mean_auc, "val/auc_median": med_auc,
        "val/auc_p25": p25_auc, "val/auc_p75": p75_auc,
        "val/auc_min": min_auc, "val/auc_max": max_auc,
        "val/faith_top1": faith_top1, "val/faith_ce": faith_ce, "epoch": 0,
    }, step=0)

    best_mean_auc = mean_auc
    ckpt_dir = ROOT / "checkpoints" / args.get("ckpt_subdir", "autoencoder_cub")
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
        mse, aucs, faith_top1, faith_ce = evaluate(
            model, h_val, z_val, y_hat_val, classifier, concept_names,
        )
        mean_auc, med_auc, p25_auc, p75_auc, min_auc, max_auc = _auc_summary(aucs)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:>3}  MSE: {mse:.6f}  "
                        f"AUC mean/median: {mean_auc:.4f}/{med_auc:.4f}  "
                        f"Faith top1: {faith_top1:.4f}  CE: {faith_ce:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "val/mse": mse, "val/auc_mean": mean_auc, "val/auc_median": med_auc,
            "val/auc_p25": p25_auc, "val/auc_p75": p75_auc,
            "val/auc_min": min_auc, "val/auc_max": max_auc,
            "val/faith_top1": faith_top1, "val/faith_ce": faith_ce,
            "train/lr": current_lr, "epoch": epoch,
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
                "class_names": class_names,
                "args": OmegaConf.to_container(args, resolve=True),
            }, ckpt_path)
            logger.info(f"  ** New best: {mean_auc:.4f} saved to {ckpt_path}")

    wandb.finish()
    logger.info(f"\nDone. Best mean AUC: {best_mean_auc:.4f}")


if __name__ == "__main__":
    main()
