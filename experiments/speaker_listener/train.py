"""
Speaker-Listener DPO training on PadChest-GR data.

Uses:
- Frozen ViT embeddings (data/padchest_final/embeddings_vit_ln.pt)
- Frozen ViT classifier head (checkpoints/padchest/classifier_vit_b16.pt, for faithfulness)
- Frozen grounded autoencoder W_E (checkpoints/autoencoder_vit/ae_ae_vit_lam0.1.pt)
- Radiologist preference pi (from data/PadChest-GR/processed_labels_gr.csv), strength tau
- Trainable Speaker MLP and Listener MLP

Loads the speaker-listener dataset (4 diagnoses) filtered by ImageID.

Usage:
    uv run experiments/speaker_listener/train.py tau=10 seed=0 run_name=l15_tau10_s0
    uv run experiments/speaker_listener/train.py literal=true tau=10 run_name=literal_tau10
"""

import copy
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

ROOT = Path(get_root_dir())
DATA_DIR = Path(os.getenv("DATA_DIR"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR"))
PADCHEST_FINAL = ROOT / "data" / "padchest_final"

# Add this dir to path so speaker.py, listener.py, reward.py can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker import Speaker, build_utterance
from listener import Listener, compute_faithfulness_targets, compute_class_concept_priors
from reward import reward, faithfulness

logger = None


def load_vit_head(device, ckpt_path: str | None = None):
    """Load the frozen ViT classifier head.

    If ckpt_path is None, falls back to the latest ViT run under
    checkpoints/padchest/ via latest.txt.
    """
    if ckpt_path is None:
        pad_ckpt_dir = ROOT / "checkpoints" / "padchest"
        vit_runs = sorted(d for d in os.listdir(pad_ckpt_dir) if d.startswith("train_cls_vit_b_16_"))
        latest_run = pad_ckpt_dir / vit_runs[-1]
        with open(latest_run / "latest.txt") as f:
            ckpt_path = f.read().strip()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    w = state["classifier.weight"]  # [K, d]
    b = state["classifier.bias"]    # [K]

    head = nn.Linear(w.shape[1], w.shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def warmup_speaker_to_literal(speaker, x_train, c_train, epochs, lr, batch_size, device):
    """Pre-train the Speaker to mimic LiteralSpeaker's per-step selection
    distribution softmax(|c|).

    Loss: cross-entropy H(p, q) = -E_p[log q] with p = softmax(|c|),
    q = softmax(speaker(x)). Equivalent to KL(p || q) up to a constant.

    After warmup, the Speaker assigns high probability to high-|c| concepts —
    matching how LiteralSpeaker samples — so DPO starts from literal's policy
    instead of random init.
    """
    speaker.train()
    optimizer = torch.optim.Adam(speaker.parameters(), lr=lr)
    N = x_train.shape[0]
    p_target = torch.softmax(c_train.abs(), dim=-1)  # [N, F]

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            log_q = torch.log_softmax(speaker(x_train[idx]), dim=-1)
            loss = -(p_target[idx] * log_q).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch == epochs or epoch % 20 == 0:
            logger.info(f"  warmup epoch {epoch:>3}/{epochs}: H(p,q) = {final_loss:.4f}")

    # Sanity check: how often does Speaker's argmax match Literal's argmax?
    speaker.eval()
    with torch.no_grad():
        speaker_top1 = speaker(x_train).argmax(dim=-1)
        literal_top1 = c_train.abs().argmax(dim=-1)
        agree = (speaker_top1 == literal_top1).float().mean().item()
    logger.info(f"  warmup done: top-1 agreement with literal = {agree:.3f}")


def load_frozen(args, device):
    # --- Autoencoder encoder weights ---
    ae_ckpt = torch.load(ROOT / args.autoencoder_ckpt, map_location="cpu", weights_only=False)
    W_E = ae_ckpt["encoder_weight"].to(device)       # [F, d]
    encoder_bias = ae_ckpt.get("encoder_bias", None)
    if encoder_bias is not None:
        encoder_bias = encoder_bias.to(device)
    n_concepts = W_E.shape[0]
    d = W_E.shape[1]
    concept_names = ae_ckpt["concept_names"]
    logger.info(f"Autoencoder: d={d}, n_concepts={n_concepts}")

    # --- ViT classifier head ---
    classifier_ckpt = args.get("classifier_ckpt", None)
    vit_head = load_vit_head(device, ckpt_path=classifier_ckpt)
    logger.info(f"ViT head: {vit_head.weight.shape[1]} -> {vit_head.weight.shape[0]}")

    # --- Embeddings ---
    dataset_dir = ROOT / "data" / args.get("dataset_dir", "padchest_final")
    emb_path = dataset_dir / args.embeddings_file
    data = torch.load(emb_path, map_location="cpu", weights_only=False)
    all_image_ids = data["ImageIDs"]
    embeddings = data["embeddings"]   # [N, d]
    z_gt = data["z_gt"]                # [N, F]
    y_hat_all = data["y_hat"]          # [N, K] — ViT predictions on original h
    sl_diagnoses = list(data["diagnosis_names"])  # [K]
    assert len(sl_diagnoses) == vit_head.weight.shape[0], (
        f"embeddings file has {len(sl_diagnoses)} diagnoses but ViT head outputs "
        f"{vit_head.weight.shape[0]} classes"
    )
    logger.info(f"Diagnoses (K={len(sl_diagnoses)}): {sl_diagnoses}")

    # --- Filter by SL dataset ImageIDs ---
    sl_df = pd.read_csv(dataset_dir / args.sl_dataset)
    sl_ids = set(sl_df["ImageID"])
    mask = torch.tensor([img_id in sl_ids for img_id in all_image_ids])
    keep_idx = mask.nonzero(as_tuple=True)[0]
    logger.info(f"SL filter: {len(keep_idx)}/{len(all_image_ids)} images kept")
    kept_image_ids = [all_image_ids[i] for i in keep_idx.tolist()]

    h = embeddings[keep_idx].to(device)
    z = z_gt[keep_idx].to(device)
    y_hat = y_hat_all[keep_idx].to(device)

    # Precompute concept activations c = h @ W_E^T (+ bias)
    if encoder_bias is not None:
        c = h @ W_E.T + encoder_bias
    else:
        c = h @ W_E.T
    logger.info(f"Concept activations: {c.shape}")

    # --- Radiologist prior from GR subset ---
    gr_df = pd.read_csv(ROOT / "data" / "PadChest-GR" / args.gr_csv)
    gr_ids = set(gr_df["ImageID"])
    # Filter z_gt and image_ids to GR subset to compute prior
    gr_kept = [img_id for img_id in all_image_ids if img_id in gr_ids]
    gr_mask = torch.tensor([img_id in gr_ids for img_id in all_image_ids])
    gr_idx = gr_mask.nonzero(as_tuple=True)[0]

    # Need diagnoses per image for prior — get from SL dataset
    # For GR images that appear in SL dataset
    sl_df_indexed = sl_df.set_index("ImageID")
    gr_in_sl = [img_id for img_id in gr_kept if img_id in sl_df_indexed.index]
    # Estimate the prior from the training split only (no validation leakage).
    split_path = dataset_dir / "split.pt"
    if split_path.exists():
        _train_ids = set(torch.load(split_path, map_location="cpu",
                                    weights_only=False)["train_image_ids"])
        gr_in_sl = [img_id for img_id in gr_in_sl if img_id in _train_ids]
    else:
        logger.warning(f"No {split_path}; estimating prior from all GR images")
    logger.info(f"GR images in SL dataset (train split): {len(gr_in_sl)}")

    # Build z_gt and diag_gt for these images
    z_gt_gr = []
    diag_gr = []
    for img_id in gr_in_sl:
        ae_idx = all_image_ids.index(img_id)
        z_gt_gr.append(z_gt[ae_idx])
        row = sl_df_indexed.loc[img_id]
        diag_gr.append([row[d] for d in sl_diagnoses])
    z_gt_gr = torch.stack(z_gt_gr)
    diag_gr = torch.tensor(diag_gr, dtype=torch.float32)
    # Map {-1, 0, 1} -> {0, 0.5, 1} for presence (we want 1.0 for present)
    diag_gr_binary = (diag_gr == 1).float()

    pi = compute_class_concept_priors(z_gt_gr, diag_gr_binary)  # [K, F]
    logger.info(f"Radiologist prior: {pi.shape}")
    logger.info("Top 5 concepts per diagnosis in prior:")
    for k, name in enumerate(sl_diagnoses):
        top5 = pi[k].argsort(descending=True)[:5]
        top5_str = ", ".join(f"{concept_names[i]}({pi[k,i]:.3f})" for i in top5)
        logger.info(f"  {name}: {top5_str}")

    return {
        "W_E": W_E,
        "encoder_bias": encoder_bias,
        "vit_head": vit_head,
        "h": h, "c": c, "z": z, "y_hat": y_hat,
        "y_gt": (data["diagnoses_gt"][keep_idx] == 1).float().to(device),  # GT diagnoses, +1 -> 1 else 0 (classifier convention)
        "n_concepts": n_concepts, "K": len(sl_diagnoses),
        "concept_names": concept_names,
        "pi": pi,
        "kept_image_ids": kept_image_ids,
        "dataset_dir": dataset_dir,
    }


def step_a_speaker_dpo(speaker, speaker_ref, listener, optimizer,
                       x, c, y_hat, W_E, vit_head, args, device,
                       iteration=1, random_frac=0.0):
    """Step A: fix listener, update speaker via DPO.

    Args:
        x: speaker input [N, input_dim] (either c or h)
        c: concept activations [N, F] (for utterance sign, faithfulness, consistency)
    """
    speaker.train()
    listener.eval()

    N = c.shape[0]
    perm = torch.randperm(N, device=device)
    total = {
        "loss": 0.0, "reward_best": 0.0, "reward_worst": 0.0,
        "utility": 0.0, "consistency": 0.0,
        "faith_best": 0.0, "faith_worst": 0.0,
        "diversity": 0.0, "lp_best": 0.0, "lp_worst": 0.0,
    }
    n_batches = 0

    for start in range(0, N, args.batch_size):
        idx = perm[start:start + args.batch_size]
        x_b = x[idx]
        c_b = c[idx]
        y_b = y_hat[idx]
        B = c_b.shape[0]

        # Logit statistics (for diversity diagnostics)
        with torch.no_grad():
            speaker_logits = speaker.forward(x_b)          # [B, F]
            speaker_logits_std = speaker_logits.std(dim=1).mean().item()
            speaker_logits_max = speaker_logits.max(dim=1).values.mean().item()
            speaker_logits_min = speaker_logits.min(dim=1).values.mean().item()
            speaker_logits_gap = speaker_logits_max - speaker_logits_min
            # Entropy of the softmax distribution
            log_probs = F.log_softmax(speaker_logits, dim=-1)
            probs = log_probs.exp()
            speaker_logits_entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        # Expand to B*b
        x_exp = x_b.unsqueeze(1).expand(-1, args.b, -1).reshape(B * args.b, -1)
        c_exp = c_b.unsqueeze(1).expand(-1, args.b, -1).reshape(B * args.b, -1)
        y_exp = y_b.unsqueeze(1).expand(-1, args.b, -1).reshape(B * args.b, -1)

        with torch.no_grad():
            indices_all, _ = speaker.sample(x_exp, args.l)  # [B*b, l]

            # Random utterances for some
            if random_frac > 0:
                n_total = B * args.b
                n_random = int(n_total * random_frac)
                if n_random > 0:
                    F_dim = c_exp.shape[1]
                    random_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
                    random_mask[torch.randperm(n_total, device=device)[:n_random]] = True
                    random_indices = torch.stack([
                        torch.randperm(F_dim, device=device)[:args.l]
                        for _ in range(n_random)
                    ])
                    indices_all[random_mask] = random_indices

            u_all = build_utterance(c_exp, indices_all)  # [B*b, F]

        # Faithfulness target via ViT head
        with torch.no_grad():
            y_tilde = compute_faithfulness_targets(u_all, c_exp, W_E.T, vit_head)

        # Reward — listener attenuation mode is configurable: sign / pos_mask / legacy.
        _mode = args.get("attenuation_mode", "sign" if args.get("sign_mode", True) else "pos_mask")
        _sign = (_mode == "sign")
        _pos_mask_kw = {"pos_mask": y_tilde} if _mode == "pos_mask" else {}
        with torch.no_grad():
            rewards = reward(u_all, listener, y_tilde, c_exp,
                             lambda_c=args.lambda_c, lambda_u=args.lambda_u,
                             sign_mode=_sign, **_pos_mask_kw)

        rewards_mat = rewards.view(B, args.b)
        indices_mat = indices_all.view(B, args.b, args.l)
        u_mat = u_all.view(B, args.b, -1)

        best_idx = rewards_mat.argmax(dim=1)
        worst_idx = rewards_mat.argmin(dim=1)
        idx_best = indices_mat[torch.arange(B, device=device), best_idx]
        idx_worst = indices_mat[torch.arange(B, device=device), worst_idx]
        u_best = u_mat[torch.arange(B, device=device), best_idx]
        u_worst = u_mat[torch.arange(B, device=device), worst_idx]

        lp_best = speaker.log_prob(x_b, idx_best)
        lp_worst = speaker.log_prob(x_b, idx_worst)
        with torch.no_grad():
            lp_ref_best = speaker_ref.log_prob(x_b, idx_best)
            lp_ref_worst = speaker_ref.log_prob(x_b, idx_worst)

        delta_theta = (lp_best - lp_ref_best) - (lp_worst - lp_ref_worst)

        with torch.no_grad():
            f_best = faithfulness(u_best, c_b, W_E, vit_head, y_b)
            f_worst = faithfulness(u_worst, c_b, W_E, vit_head, y_b)
        delta_faith = f_best - f_worst

        beta_eff = args.beta if iteration > 1 else 0.0
        dpo_logit = beta_eff * delta_theta - args.gamma_faith * delta_faith
        loss = -F.logsigmoid(dpo_logit).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total["loss"] += loss.item()
        total["reward_best"] += rewards_mat[torch.arange(B, device=device), best_idx].mean().item()
        total["reward_worst"] += rewards_mat[torch.arange(B, device=device), worst_idx].mean().item()
        total["faith_best"] += f_best.mean().item()
        total["faith_worst"] += f_worst.mean().item()
        total["lp_best"] += lp_best.mean().item()
        total["lp_worst"] += lp_worst.mean().item()

        # Diversity: unique utterance sets per image
        with torch.no_grad():
            sorted_idx = indices_mat.sort(dim=-1).values
            unique_counts = [
                len({tuple(sorted_idx[i, j].tolist()) for j in range(args.b)})
                for i in range(B)
            ]
            total["diversity"] += sum(unique_counts) / B

        # Decompose reward
        with torch.no_grad():
            yt_best = y_tilde.view(B, args.b, -1)[torch.arange(B, device=device), best_idx]
            _pm_best = {"pos_mask": yt_best} if _mode == "pos_mask" else {}
            output = listener(u_best, sign_mode=_sign, **_pm_best)
            if listener.tau > 0 and listener.pi is not None:
                eps = 1e-7
                logp = (y_tilde.view(B, args.b, -1)[torch.arange(B, device=device), best_idx]
                        * output.clamp(min=eps).log()
                        + (1 - y_tilde.view(B, args.b, -1)[torch.arange(B, device=device), best_idx])
                        * (1 - output).clamp(min=eps).log()).mean(dim=1)
            else:
                logp = -F.binary_cross_entropy_with_logits(
                    output,
                    y_tilde.view(B, args.b, -1)[torch.arange(B, device=device), best_idx],
                    reduction="none",
                ).mean(dim=1)
            total["utility"] += logp.mean().item()

            m_u_best = (u_best != 0).float()
            cons = -F.binary_cross_entropy_with_logits(
                c_b.abs(), m_u_best, reduction="none"
            ).mean(dim=1)
            total["consistency"] += cons.mean().item()

        n_batches += 1

        # Per-batch wandb logging (under train/)
        wandb.log({
            "train/dpo_loss": loss.item(),
            "train/reward_best": rewards_mat[torch.arange(B, device=device), best_idx].mean().item(),
            "train/reward_worst": rewards_mat[torch.arange(B, device=device), worst_idx].mean().item(),
            "train/reward_gap": (rewards_mat[torch.arange(B, device=device), best_idx]
                                  - rewards_mat[torch.arange(B, device=device), worst_idx]).mean().item(),
            "train/lp_best": lp_best.mean().item(),
            "train/lp_worst": lp_worst.mean().item(),
            "train/faith_best": f_best.mean().item(),
            "train/faith_worst": f_worst.mean().item(),
            "train/diversity": sum(unique_counts) / B,
            "train/utility": logp.mean().item(),
            "train/consistency": cons.mean().item(),
            "train/speaker_logits_std": speaker_logits_std,
            "train/speaker_logits_max": speaker_logits_max,
            "train/speaker_logits_min": speaker_logits_min,
            "train/speaker_logits_gap": speaker_logits_gap,
            "train/speaker_logits_entropy": speaker_logits_entropy,
        })

    return {k: v / max(n_batches, 1) for k, v in total.items()}


def step_b_listener(speaker, listener, listener_optimizer,
                    x, c, W_E, vit_head, args, device, pos_weight=None):
    """Step B: fix speaker, update listener.

    Args:
        x: speaker input [N, input_dim] (c or h)
        c: concept activations [N, F] (for utterance and faithfulness)
        pos_weight: optional [K] per-class positive weight for class-imbalanced BCE
    """
    speaker.eval()
    listener.train()

    N = c.shape[0]
    total_loss = 0.0
    n_batches = 0

    for _ in range(args.listener_epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, args.batch_size):
            idx = perm[start:start + args.batch_size]
            x_b = x[idx]
            c_b = c[idx]
            B = c_b.shape[0]

            x_exp = x_b.unsqueeze(1).expand(-1, args.n_expl, -1).reshape(B * args.n_expl, -1)
            c_exp = c_b.unsqueeze(1).expand(-1, args.n_expl, -1).reshape(B * args.n_expl, -1)
            with torch.no_grad():
                indices_all, _ = speaker.sample(x_exp, args.l)
                u_all = build_utterance(c_exp, indices_all)
                y_tilde = compute_faithfulness_targets(u_all, c_exp, W_E.T, vit_head)

            _mode_b = args.get("attenuation_mode", "sign" if args.get("sign_mode", True) else "pos_mask")
            _sign_b = (_mode_b == "sign")
            _pm_b = {"pos_mask": y_tilde} if _mode_b == "pos_mask" else {}
            output = listener(u_all, sign_mode=_sign_b, **_pm_b)
            if listener.tau > 0 and listener.pi is not None:
                eps = 1e-7
                w = pos_weight.view(1, -1) if pos_weight is not None else 1.0
                loss = -(w * y_tilde * output.clamp(min=eps).log()
                         + (1 - y_tilde) * (1 - output).clamp(min=eps).log()).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(output, y_tilde, pos_weight=pos_weight)

            listener_optimizer.zero_grad()
            loss.backward()
            listener_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return {"listener_loss": total_loss / max(n_batches, 1)}


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "configs"),
    config_name="speaker_listener",
)
def main(args: DictConfig):
    global logger
    import hydra as _h
    hydra_cfg = _h.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg.runtime.output_dir
    run_name = args.run_name or "sl_default"
    logger = setup_logger(log_dir=log_dir, filename=f"{run_name}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {OmegaConf.to_yaml(args)}")

    frozen = load_frozen(args, device)
    W_E = frozen["W_E"]
    vit_head = frozen["vit_head"]
    h = frozen["h"]
    c = frozen["c"]
    z = frozen["z"]
    y_hat = frozen["y_hat"]
    n_concepts = frozen["n_concepts"]
    K = frozen["K"]
    pi = frozen["pi"]
    concept_names = frozen["concept_names"]
    d = h.shape[1]  # ViT embedding dim

    # Speaker input selection. LiteralSpeaker always reads c regardless of config.
    if args.get("literal", False):
        speaker_x = c
        speaker_input_dim = n_concepts
        logger.info(f"Literal baseline: deterministic top-l by |c|, no DPO training")
    elif args.speaker_input == "embedding":
        speaker_x = h
        speaker_input_dim = d
        logger.info(f"Speaker input: image embedding (d={d})")
    elif args.speaker_input == "concepts":
        speaker_x = c
        speaker_input_dim = n_concepts
        logger.info(f"Speaker input: concept activations (F={n_concepts})")
    else:
        raise ValueError(f"Unknown speaker_input: {args.speaker_input}")

    # Train/val split — prefer the pre-computed stratified split.pt if present;
    # fall back to a random split otherwise.
    N = c.shape[0]
    kept_image_ids = frozen["kept_image_ids"]
    split_path = frozen["dataset_dir"] / "split.pt"
    if split_path.exists():
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        train_ids = set(split["train_image_ids"])
        val_ids = set(split["val_image_ids"])
        # Map filtered positions to train/val based on ImageID
        train_idx = torch.tensor(
            [i for i, iid in enumerate(kept_image_ids) if iid in train_ids],
            dtype=torch.long, device=device,
        )
        val_idx = torch.tensor(
            [i for i, iid in enumerate(kept_image_ids) if iid in val_ids],
            dtype=torch.long, device=device,
        )
        unmapped = N - (len(train_idx) + len(val_idx))
        logger.info(f"Loaded split from {split_path}: train={len(train_idx)}, val={len(val_idx)}"
                    + (f", unmapped={unmapped}" if unmapped else ""))
    else:
        logger.info(f"No {split_path}; using random split (seed={args.seed})")
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(N, generator=g)
        n_train = int(N * args.train_ratio)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    x_train = speaker_x[train_idx]; x_val = speaker_x[val_idx]
    # PCBM-style fixed speaker: replace the selection score |c| with the
    # surrogate contribution |A[y_hat]*c| (A = L1 logistic fit on train split).
    # Requires literal=true (fixed speaker, no DPO). Signs/targets still use c.
    if args.get("speaker_mode", None) == "pcbm":
        assert args.get("literal", False), "speaker_mode=pcbm requires literal=true"
        from pcbm import fit_pcbm_head, pcbm_scores
        A_pcbm = fit_pcbm_head(c[train_idx], frozen["y_gt"][train_idx], multilabel=True)  # canonical PCBM: fit on ground truth
        speaker_x = pcbm_scores(c, y_hat, A_pcbm, multilabel=True)
        x_train = speaker_x[train_idx]; x_val = speaker_x[val_idx]
        logger.info(f"PCBM speaker: top-l by |A[y_hat]*c|, A nnz={int((A_pcbm!=0).sum())}/{A_pcbm.size} (fit on train split)")
    c_train = c[train_idx]; c_val = c[val_idx]
    y_train = y_hat[train_idx]; y_val = y_hat[val_idx]

    # Per-class pos_weight for BCE (handles diagnosis-frequency imbalance in listener training)
    pos_freq = y_train.mean(dim=0).clamp(min=0.01, max=0.99)       # [K]
    listener_pos_weight = ((1 - pos_freq) / pos_freq).to(device)   # [K]
    logger.info(f"Listener pos_weight per class: {listener_pos_weight.tolist()}")

    # Initialize speaker and listener
    if args.get("literal", False):
        from speaker import LiteralSpeaker
        speaker = LiteralSpeaker(n_concepts).to(device)
    else:
        speaker = Speaker(n_concepts, input_dim=speaker_input_dim).to(device)
        # Optional: warm-start the Speaker by imitating LiteralSpeaker's
        # selection distribution softmax(|c|). This makes the DPO comparison
        # against literal a "behavior-clone + RL refinement" — DPO starts at
        # literal's policy and only has to improve.
        if args.get("init_from_literal", False):
            warmup_speaker_to_literal(
                speaker=speaker,
                x_train=x_train,
                c_train=c_train,
                epochs=int(args.get("init_epochs", 100)),
                lr=float(args.get("init_lr", 1e-3)),
                batch_size=int(args.batch_size),
                device=device,
            )
    listener = Listener(
        n_concepts, K,
        logit_clamp=float(args.get("logit_clamp", 5.0)),
    ).to(device)
    if args.tau > 0:
        listener.set_priors(pi.to(device), tau=args.tau)
        logger.info(f"Listener with coverage preference (tau={args.tau})")

    speaker_ref = copy.deepcopy(speaker)
    speaker_ref.eval()
    for p in speaker_ref.parameters():
        p.requires_grad_(False)

    speaker_opt = torch.optim.AdamW(speaker.parameters(), lr=args.lr_speaker)
    listener_opt = torch.optim.AdamW(listener.parameters(), lr=args.lr_listener)

    # Optional: pre-train listener on utterances from current speaker (which is
    # at literal-equivalent distribution after warmup). Gives the listener a
    # meaningful starting state so iter-1 utility reward isn't pure noise.
    if args.get("init_listener_with_literal", False):
        logger.info(
            f"Pre-training listener on literal-style utterances "
            f"({args.get('init_listener_epochs', 1)} epoch(s))"
        )
        n_epochs_orig = args.listener_epochs
        from omegaconf import open_dict
        with open_dict(args):
            args.listener_epochs = int(args.get("init_listener_epochs", 1))
        result = step_b_listener(
            speaker=speaker, listener=listener, listener_optimizer=listener_opt,
            x=x_train, c=c_train, W_E=W_E, vit_head=vit_head,
            args=args, device=device, pos_weight=listener_pos_weight,
        )
        with open_dict(args):
            args.listener_epochs = n_epochs_orig
        logger.info(f"  listener pre-train loss: {result['listener_loss']:.4f}")

    # Wandb
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(args, resolve=True),
    )

    for t in range(1, args.T + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {t}/{args.T}")
        logger.info(f"{'='*60}")

        # Random warmup schedule: linear decay from args.random_warmup to 0 over full T
        if args.random_warmup > 0:
            random_frac = args.random_warmup * max(0.0, 1 - (t - 1) / max(args.T - 1, 1))
        else:
            random_frac = 0.0

        if args.get("literal", False):
            logger.info("Step A: skipped (literal speaker is untrained)")
            speaker_metrics = {}
        else:
            logger.info(f"Step A: Speaker DPO (random_frac={random_frac:.2f})")
            speaker_metrics = step_a_speaker_dpo(
                speaker, speaker_ref, listener, speaker_opt,
                x_train, c_train, y_train, W_E, vit_head, args, device,
                iteration=t, random_frac=random_frac,
            )
            speaker_ref = copy.deepcopy(speaker)
            speaker_ref.eval()
            for p in speaker_ref.parameters():
                p.requires_grad_(False)

        for k, v in speaker_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        logger.info(f"Step B: Listener")
        listener_metrics = step_b_listener(
            speaker, listener, listener_opt,
            x_train, c_train, W_E, vit_head, args, device,
            pos_weight=listener_pos_weight,
        )
        logger.info(f"  listener_loss: {listener_metrics['listener_loss']:.4f}")

        # Val (deterministic: take top-l by speaker logits)
        speaker.eval(); listener.eval()
        with torch.no_grad():
            val_logits = speaker.forward(x_val)
            val_indices = val_logits.topk(args.l, dim=1).indices
            val_u = build_utterance(c_val, val_indices)
            val_y_tilde = compute_faithfulness_targets(val_u, c_val, W_E.T, vit_head)
            _mode_v = args.get("attenuation_mode", "sign" if args.get("sign_mode", True) else "pos_mask")
            _sign_v = (_mode_v == "sign")
            _pm_v = {"pos_mask": val_y_tilde} if _mode_v == "pos_mask" else {}
            val_rewards = reward(val_u, listener, val_y_tilde, c_val,
                                 lambda_c=args.lambda_c, lambda_u=args.lambda_u,
                                 sign_mode=_sign_v, **_pm_v)

            # Balanced-accuracy helper: mean per-class (TPR+TNR)/2, averaged over classes
            # with both positives and negatives. Constant predictor = 0.5, perfect = 1.0.
            def _balanced_acc(preds, target):
                tp = ((preds == 1) & (target == 1)).float().sum(dim=0)
                tn = ((preds == 0) & (target == 0)).float().sum(dim=0)
                pos = target.sum(dim=0)
                neg = (1 - target).sum(dim=0)
                tpr = tp / pos.clamp(min=1)
                tnr = tn / neg.clamp(min=1)
                bal = 0.5 * (tpr + tnr)
                valid = ((pos > 0) & (neg > 0))
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    return float("nan"), 0
                return bal[valid].mean().item(), n_valid

            # Faithfulness as balanced accuracy between reconstructed and original ViT preds.
            m_u = (val_u != 0).float()
            h_hat = (m_u * c_val) @ W_E
            p_recon = torch.sigmoid(vit_head(h_hat))        # [N, K]
            preds_recon = (p_recon > 0.5).float()
            val_faith_agree, n_valid_faith = _balanced_acc(preds_recon, y_val)

            output = listener(val_u, sign_mode=_sign_v, **_pm_v)
            if listener.tau > 0 and listener.pi is not None:
                eps = 1e-7
                w = listener_pos_weight.view(1, -1)
                val_ll = -(w * val_y_tilde * output.clamp(min=eps).log()
                           + (1 - val_y_tilde) * (1 - output).clamp(min=eps).log()).mean().item()
                listener_probs = output
            else:
                val_ll = F.binary_cross_entropy_with_logits(
                    output, val_y_tilde, pos_weight=listener_pos_weight
                ).item()
                listener_probs = torch.sigmoid(output)

            # Balanced listener accuracy vs two targets:
            # - recon: ỹ(u) = ViT on masked reconstruction (listener's training target)
            # - orig:  ŷ    = ViT on full h (ground-truth ViT prediction)
            # Balanced acc = mean of (TPR + TNR) / 2 per diagnosis, averaged over K
            # classes that have both positives and negatives. Constant predictor = 0.5.
            listener_preds = (listener_probs > 0.5).float()
            y_tilde_binary = (val_y_tilde > 0.5).float()
            val_listener_acc_recon, n_valid_recon = _balanced_acc(listener_preds, y_tilde_binary)
            val_listener_acc_orig, n_valid_orig = _balanced_acc(listener_preds, y_val)

            # Consistency agreement: fraction of speaker's l selections that are in AE top-2l by |c|.
            # Target set is 2l to tolerate near-ties in the long tail of |c|.
            # Random baseline = 2l / F.
            F_dim = c_val.shape[1]
            k_target = min(2 * args.l, F_dim)
            topk_idx = c_val.abs().topk(k_target, dim=1).indices
            top_mask = torch.zeros_like(c_val).scatter_(1, topk_idx, 1.0)
            val_consistency_agree = ((m_u * top_mask).sum(dim=1) / args.l).mean().item()

            # Unique utterance count across val set — diagnostic for per-image mode collapse
            val_unique_utterances = len({tuple(row.tolist()) for row in val_u})

        wandb.log({
            "val/reward": val_rewards.mean().item(),
            "val/faith_agree": val_faith_agree,
            "val/consistency_agree": val_consistency_agree,
            "val/listener_loss": val_ll,
            "val/listener_acc_recon": val_listener_acc_recon,
            "val/listener_acc_orig": val_listener_acc_orig,
            "val/unique_utterances": val_unique_utterances,
            "iteration": t,
        })
        logger.info(f"  val/reward: {val_rewards.mean().item():.4f}")
        logger.info(f"  val/faith_agree: {val_faith_agree:.4f}  (n_valid_classes = {n_valid_faith}/{y_val.shape[1]})")
        logger.info(f"  val/consistency_agree: {val_consistency_agree:.4f}  (random = {k_target/c_val.shape[1]:.3f})")
        logger.info(f"  val/listener_loss: {val_ll:.4f}")
        logger.info(f"  val/listener_acc_recon: {val_listener_acc_recon:.4f}  (n_valid_classes = {n_valid_recon}/{val_y_tilde.shape[1]})")
        logger.info(f"  val/listener_acc_orig:  {val_listener_acc_orig:.4f}  (n_valid_classes = {n_valid_orig}/{val_y_tilde.shape[1]})")
        logger.info(f"  val/unique_utterances: {val_unique_utterances}  (N_val = {val_u.shape[0]})")

        # Checkpoint — controlled by args.save_checkpoint (default False).
        if args.get("save_checkpoint", False):
            ckpt_subdir = args.get("ckpt_subdir", "speaker_listener")
            ckpt_dir = ROOT / "checkpoints" / ckpt_subdir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "iteration": t,
                "speaker_state_dict": speaker.state_dict(),
                "listener_state_dict": listener.state_dict(),
                "args": OmegaConf.to_container(args, resolve=True),
                "n_concepts": n_concepts,
                "K": K,
                "concept_names": concept_names,
            }, ckpt_dir / f"sl_{run_name}_iter{t}.pt")

    wandb.finish()
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
