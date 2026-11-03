"""
Speaker-Listener DPO training on CUB (single-label, 200 species).

Uses:
- Frozen CLIP-ViT-L/14 image embeddings (data/CUB/embeddings_clip_50.pt)
- Frozen MLP classifier head (checkpoints/cub_mlp/classifier_mlp_50.pt)
- Frozen autoencoder W_E (checkpoints/autoencoder_cub/ae_ae_cub_lam0.1.pt)
- Listener preference pi (data/CUB/cub_class_attribute_prior_50.pt), strength tau
- Trainable Speaker MLP and Listener MLP

Train/val split: from `is_train` field in the embeddings file (the official
80/20 CUB split).

Usage:
    uv run experiments/cub/train_speaker_listener.py
    uv run experiments/cub/train_speaker_listener.py literal=true run_name=literal
    uv run experiments/cub/train_speaker_listener.py l=10 lambda_c=0.0 run_name=l10_lc0
"""

import copy
import os
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf

from experiments.cub.train_mlp_classifier import MLPClassifier
from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

ROOT = Path(get_root_dir())

# Reuse Speaker / LiteralSpeaker / build_utterance from the PadChest module —
# they're class-agnostic and only depend on F.
sys.path.insert(0, str(ROOT / "experiments" / "speaker_listener"))
from speaker import Speaker, LiteralSpeaker, build_utterance  # noqa: E402

# CUB-local listener and reward (single-label).
from experiments.cub.listener import (  # noqa: E402
    Listener, compute_faithfulness_targets, listener_loss,
)
from experiments.cub.reward import reward, faithfulness  # noqa: E402

logger = None


def load_classifier(ckpt_path: Path, device):
    """Load the frozen MLP classifier head trained on CLIP embeddings."""
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
        f"MLP classifier: {ckpt['input_dim']} -> "
        f"{'->'.join(map(str, ckpt['hidden_dims']))} -> {ckpt['output_dim']}  "
        f"(val_top1={ckpt.get('val_top1', float('nan')):.4f})"
    )
    return model


def warmup_speaker_to_literal(speaker, x_train, c_train, epochs, lr, batch_size,
                              device):
    """Behavior-clone LiteralSpeaker's selection distribution softmax(|c|)."""
    speaker.train()
    optimizer = torch.optim.Adam(speaker.parameters(), lr=lr)
    N = x_train.shape[0]
    p_target = torch.softmax(c_train.abs(), dim=-1)

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

    speaker.eval()
    with torch.no_grad():
        speaker_top1 = speaker(x_train).argmax(dim=-1)
        literal_top1 = c_train.abs().argmax(dim=-1)
        agree = (speaker_top1 == literal_top1).float().mean().item()
    logger.info(f"  warmup done: top-1 agreement with literal = {agree:.3f}")


def load_frozen(args, device):
    # --- Autoencoder encoder weights ---
    ae_ckpt = torch.load(ROOT / args.autoencoder_ckpt, map_location="cpu",
                         weights_only=False)
    W_E = ae_ckpt["encoder_weight"].to(device)            # [F, d]
    encoder_bias = ae_ckpt.get("encoder_bias", None)
    if encoder_bias is not None:
        encoder_bias = encoder_bias.to(device)
    n_concepts = W_E.shape[0]
    d = W_E.shape[1]
    concept_names = ae_ckpt["concept_names"]
    logger.info(f"Autoencoder: d={d}, n_concepts={n_concepts}")

    # --- MLP classifier head ---
    classifier = load_classifier(ROOT / args.classifier_ckpt, device)
    K = classifier.net[-1].out_features
    logger.info(f"K = {K} classes")

    # --- Embeddings (filtered by is_train) ---
    dataset_dir = ROOT / "data" / args.get("dataset_dir", "CUB")
    emb_path = dataset_dir / args.embeddings_file
    data = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]                       # [N, d]
    z_gt = data["z_gt"]                                   # [N, F]
    is_train = data["is_train"].bool()                    # [N]
    class_names = data["class_names"]                     # [K]
    assert embeddings.shape[1] == d, (
        f"Embedding dim {embeddings.shape[1]} != AE input dim {d}"
    )
    assert len(class_names) == K, (
        f"class_names has {len(class_names)} entries but classifier outputs {K}"
    )

    h = embeddings.to(device)
    z = z_gt.to(device)

    # Concept activations c = h @ W_E.T (+ bias)
    if encoder_bias is not None:
        c = h @ W_E.T + encoder_bias
    else:
        c = h @ W_E.T
    logger.info(f"Concept activations: {c.shape}")

    # y_hat = classifier argmax on the original h (integer class).
    with torch.no_grad():
        y_hat = classifier(h).argmax(dim=-1).long()       # [N]

    return {
        "W_E": W_E, "encoder_bias": encoder_bias,
        "classifier": classifier,
        "h": h, "c": c, "z": z, "y_hat": y_hat,
        "y_gt": (data["class_id"].long() - 1).to(device),   # GT class index (class_id is 1-based)
        "is_train": is_train.to(device),
        "n_concepts": n_concepts, "K": K,
        "concept_names": concept_names, "class_names": class_names,
    }


def step_a_speaker_dpo(speaker, speaker_ref, listener, optimizer,
                       x, c, y_hat, W_E, classifier, args, device,
                       iteration=1, random_frac=0.0):
    """Step A: fix listener, update speaker via DPO."""
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
        y_b = y_hat[idx]                                  # [B] integer class
        B = c_b.shape[0]

        with torch.no_grad():
            speaker_logits = speaker.forward(x_b)
            speaker_logits_std = speaker_logits.std(dim=1).mean().item()
            speaker_logits_max = speaker_logits.max(dim=1).values.mean().item()
            speaker_logits_min = speaker_logits.min(dim=1).values.mean().item()
            speaker_logits_gap = speaker_logits_max - speaker_logits_min
            log_probs = F.log_softmax(speaker_logits, dim=-1)
            probs = log_probs.exp()
            speaker_logits_entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        # Expand to B*b
        x_exp = x_b.unsqueeze(1).expand(-1, args.b, -1).reshape(B * args.b, -1)
        c_exp = c_b.unsqueeze(1).expand(-1, args.b, -1).reshape(B * args.b, -1)
        y_exp = y_b.unsqueeze(1).expand(-1, args.b).reshape(B * args.b)

        with torch.no_grad():
            indices_all, _ = speaker.sample(x_exp, args.l)            # [B*b, l]

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

            u_all = build_utterance(c_exp, indices_all)               # [B*b, F]

        with torch.no_grad():
            y_tilde = compute_faithfulness_targets(u_all, c_exp, W_E.T, classifier)

        _mode = args.get("attenuation_mode", "sign" if args.get("sign_mode", True) else "pos_mask")
        _sign = (_mode == "sign")
        K_total = y_tilde.shape[-1]
        if _mode == "pos_mask":
            _pm = {"pos_mask": y_tilde}
        elif _mode == "target_only":
            # target = argmax of soft y_tilde (per-utterance, dynamic)
            target = y_tilde.argmax(dim=-1)
            _pm = {"pos_mask": F.one_hot(target, num_classes=K_total).float()}
        else:
            _pm = {}
        with torch.no_grad():
            rewards = reward(u_all, listener, y_tilde, c_exp,
                             lambda_c=args.lambda_c, lambda_u=args.lambda_u,
                             sign_mode=_sign, **_pm)

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
            f_best = faithfulness(u_best, c_b, W_E, classifier, y_b)
            f_worst = faithfulness(u_worst, c_b, W_E, classifier, y_b)
        delta_faith = f_best - f_worst

        beta_eff = args.beta if iteration > 1 else 0.0
        dpo_logit = beta_eff * delta_theta - args.gamma_faith * delta_faith
        loss = -F.logsigmoid(dpo_logit).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total["loss"] += loss.item()
        total["reward_best"] += rewards_mat[torch.arange(B, device=device), best_idx].mean().item()
        total["reward_worst"] += rewards_mat[torch.arange(B, device=device), worst_idx].mean().item()
        total["faith_best"] += f_best.mean().item()
        total["faith_worst"] += f_worst.mean().item()
        total["lp_best"] += lp_best.mean().item()
        total["lp_worst"] += lp_worst.mean().item()

        with torch.no_grad():
            sorted_idx = indices_mat.sort(dim=-1).values
            unique_counts = [
                len({tuple(sorted_idx[i, j].tolist()) for j in range(args.b)})
                for i in range(B)
            ]
            total["diversity"] += sum(unique_counts) / B

            # Decompose reward: utility (soft CE) + consistency
            yt_best = y_tilde.view(B, args.b, -1)[torch.arange(B, device=device), best_idx]
            if _mode == "pos_mask":
                _pm_best = {"pos_mask": yt_best}
            elif _mode == "target_only":
                target_best = yt_best.argmax(dim=-1)
                _pm_best = {"pos_mask": F.one_hot(target_best, num_classes=K_total).float()}
            else:
                _pm_best = {}
            logits_lst = listener(u_best, sign_mode=_sign, **_pm_best)
            log_p_lst = F.log_softmax(logits_lst, dim=-1)
            util = (yt_best * log_p_lst).sum(dim=-1)
            total["utility"] += util.mean().item()

            m_u_best = (u_best != 0).float()
            cons = (m_u_best * F.logsigmoid(c_b.abs())).sum(dim=1) \
                / m_u_best.sum(dim=1).clamp(min=1)
            total["consistency"] += cons.mean().item()

        n_batches += 1

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
            "train/utility": util.mean().item(),
            "train/consistency": cons.mean().item(),
            "train/speaker_logits_std": speaker_logits_std,
            "train/speaker_logits_max": speaker_logits_max,
            "train/speaker_logits_min": speaker_logits_min,
            "train/speaker_logits_gap": speaker_logits_gap,
            "train/speaker_logits_entropy": speaker_logits_entropy,
        })

    return {k: v / max(n_batches, 1) for k, v in total.items()}


def step_b_listener(speaker, listener, listener_optimizer,
                    x, c, W_E, classifier, args, device, y_hat=None):
    """Step B: fix speaker, update listener (single-label soft CE).

    y_hat (optional [N] integer): per-image target class (= argmax classifier(h)),
    used to build a one-hot pos_mask when attenuation_mode='target_only'."""
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
            y_b = y_hat[idx] if y_hat is not None else None

            x_exp = x_b.unsqueeze(1).expand(-1, args.n_expl, -1).reshape(B * args.n_expl, -1)
            c_exp = c_b.unsqueeze(1).expand(-1, args.n_expl, -1).reshape(B * args.n_expl, -1)
            with torch.no_grad():
                indices_all, _ = speaker.sample(x_exp, args.l)
                u_all = build_utterance(c_exp, indices_all)
                y_tilde = compute_faithfulness_targets(u_all, c_exp, W_E.T, classifier)

            _mode_b = args.get("attenuation_mode", "sign" if args.get("sign_mode", True) else "pos_mask")
            _sign_b = (_mode_b == "sign")
            K_total = y_tilde.shape[-1]
            if _mode_b == "pos_mask":
                _pm_b = {"pos_mask": y_tilde}
            elif _mode_b == "target_only":
                target_b = y_tilde.argmax(dim=-1)
                _pm_b = {"pos_mask": F.one_hot(target_b, num_classes=K_total).float()}
            else:
                _pm_b = {}
            loss = listener_loss(listener, u_all, y_tilde, sign_mode=_sign_b, **_pm_b)

            listener_optimizer.zero_grad()
            loss.backward()
            listener_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return {"listener_loss": total_loss / max(n_batches, 1)}


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "configs"),
    config_name="speaker_listener_cub",
)
def main(args: DictConfig):
    global logger
    import hydra as _h
    hydra_cfg = _h.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg.runtime.output_dir
    run_name = args.run_name or "sl_cub_default"
    logger = setup_logger(log_dir=log_dir, filename=f"{run_name}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {OmegaConf.to_yaml(args)}")

    frozen = load_frozen(args, device)
    W_E = frozen["W_E"]
    classifier = frozen["classifier"]
    h = frozen["h"]
    c = frozen["c"]
    y_hat = frozen["y_hat"]
    is_train = frozen["is_train"]
    n_concepts = frozen["n_concepts"]
    K = frozen["K"]
    concept_names = frozen["concept_names"]
    d = h.shape[1]

    # Speaker input. LiteralSpeaker always reads c.
    if args.get("literal", False):
        speaker_x = c
        speaker_input_dim = n_concepts
        logger.info("Literal baseline: deterministic top-l by |c|, no DPO training")
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

    # Train/val split from is_train (the official 80/20 CUB split).
    train_idx = is_train.nonzero(as_tuple=True)[0]
    val_idx = (~is_train).nonzero(as_tuple=True)[0]
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    x_train = speaker_x[train_idx]; x_val = speaker_x[val_idx]
    # PCBM-style fixed speaker (see experiments/speaker_listener/pcbm.py).
    if args.get("speaker_mode", None) == "pcbm":
        assert args.get("literal", False), "speaker_mode=pcbm requires literal=true"
        from pcbm import fit_pcbm_head, pcbm_scores
        A_pcbm = fit_pcbm_head(c[train_idx], frozen["y_gt"][train_idx], multilabel=False)  # canonical PCBM: fit on ground truth
        speaker_x = pcbm_scores(c, y_hat, A_pcbm, multilabel=False)
        x_train = speaker_x[train_idx]; x_val = speaker_x[val_idx]
        logger.info(f"PCBM speaker: top-l by |A[y_hat]*c|, A nnz={int((A_pcbm!=0).sum())}/{A_pcbm.size} (fit on train split)")
    c_train = c[train_idx]; c_val = c[val_idx]
    y_train = y_hat[train_idx]; y_val = y_hat[val_idx]

    # Speaker / listener init
    if args.get("literal", False):
        speaker = LiteralSpeaker(n_concepts).to(device)
    else:
        speaker = Speaker(n_concepts, input_dim=speaker_input_dim).to(device)
        if args.get("init_from_literal", False):
            warmup_speaker_to_literal(
                speaker=speaker, x_train=x_train, c_train=c_train,
                epochs=int(args.get("init_epochs", 100)),
                lr=float(args.get("init_lr", 1e-3)),
                batch_size=int(args.batch_size), device=device,
            )

    listener = Listener(
        n_concepts, K,
        logit_clamp=float(args.get("logit_clamp", 5.0)),
    ).to(device)

    # Load class-attribute prior (from build_cub_prior.py) and attach to listener
    # iff tau > 0.
    tau = float(args.get("tau", 0))
    if tau > 0:
        prior_path = ROOT / args.get(
            "prior_path", "data/CUB/cub_class_attribute_prior.pt")
        prior = torch.load(prior_path, map_location=device, weights_only=False)
        pi = prior["pi"].to(device)                               # [200, 312]
        listener.set_priors(pi, tau=tau)
        logger.info(f"Listener priors set: pi {tuple(pi.shape)}, tau={tau}, "
                    f"mode={args.get('attenuation_mode', 'sign')}")

    speaker_ref = copy.deepcopy(speaker)
    speaker_ref.eval()
    for p in speaker_ref.parameters():
        p.requires_grad_(False)

    speaker_opt = torch.optim.AdamW(speaker.parameters(), lr=args.lr_speaker)
    listener_opt = torch.optim.AdamW(listener.parameters(), lr=args.lr_listener)

    if args.get("init_listener_with_literal", False):
        logger.info(
            f"Pre-training listener on literal-style utterances "
            f"({args.get('init_listener_epochs', 1)} epoch(s))"
        )
        from omegaconf import open_dict
        n_epochs_orig = args.listener_epochs
        with open_dict(args):
            args.listener_epochs = int(args.get("init_listener_epochs", 1))
        result = step_b_listener(
            speaker=speaker, listener=listener, listener_optimizer=listener_opt,
            x=x_train, c=c_train, W_E=W_E, classifier=classifier,
            args=args, device=device, y_hat=y_train,
        )
        with open_dict(args):
            args.listener_epochs = n_epochs_orig
        logger.info(f"  listener pre-train loss: {result['listener_loss']:.4f}")

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(args, resolve=True),
    )

    for t in range(1, args.T + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {t}/{args.T}")
        logger.info(f"{'='*60}")

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
                x_train, c_train, y_train, W_E, classifier, args, device,
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
            x_train, c_train, W_E, classifier, args, device, y_hat=y_train,
        )
        logger.info(f"  listener_loss: {listener_metrics['listener_loss']:.4f}")

        # Val (deterministic: top-l by speaker logits)
        speaker.eval(); listener.eval()
        with torch.no_grad():
            val_logits = speaker.forward(x_val)
            val_indices = val_logits.topk(args.l, dim=1).indices
            val_u = build_utterance(c_val, val_indices)
            val_y_tilde = compute_faithfulness_targets(val_u, c_val, W_E.T, classifier)
            _mode_v = args.get("attenuation_mode", "sign" if args.get("sign_mode", True) else "pos_mask")
            _sign_v = (_mode_v == "sign")
            K_total = val_y_tilde.shape[-1]
            if _mode_v == "pos_mask":
                _pm_v = {"pos_mask": val_y_tilde}
            elif _mode_v == "target_only":
                val_target = val_y_tilde.argmax(dim=-1)
                _pm_v = {"pos_mask": F.one_hot(val_target, num_classes=K_total).float()}
            else:
                _pm_v = {}
            val_rewards = reward(val_u, listener, val_y_tilde, c_val,
                                 lambda_c=args.lambda_c, lambda_u=args.lambda_u,
                                 sign_mode=_sign_v, **_pm_v)

            # Faithfulness: top-1 agreement between argmax(classifier(h_hat))
            # and y_val (= classifier argmax on h, the original embedding).
            m_u = (val_u != 0).float()
            h_hat = (m_u * c_val) @ W_E
            recon_logits = classifier(h_hat)
            preds_recon = recon_logits.argmax(dim=-1)
            val_faith_top1 = (preds_recon == y_val).float().mean().item()
            _, recon_top3 = recon_logits.topk(3, dim=-1)
            _, recon_top5 = recon_logits.topk(5, dim=-1)
            val_faith_top3 = (recon_top3 == y_val.unsqueeze(1)).any(dim=-1).float().mean().item()
            val_faith_top5 = (recon_top5 == y_val.unsqueeze(1)).any(dim=-1).float().mean().item()

            # Listener accuracy vs two targets:
            # - recon: argmax of y_tilde (listener's training target)
            # - orig:  y_val (= argmax of classifier on h)
            listener_logits = listener(val_u, sign_mode=_sign_v, **_pm_v)
            listener_pred = listener_logits.argmax(dim=-1)
            yt_argmax = val_y_tilde.argmax(dim=-1)
            val_listener_acc_recon = (listener_pred == yt_argmax).float().mean().item()
            val_listener_acc_orig = (listener_pred == y_val).float().mean().item()
            _, lst_top3 = listener_logits.topk(3, dim=-1)
            _, lst_top5 = listener_logits.topk(5, dim=-1)
            val_listener_top3_orig = (lst_top3 == y_val.unsqueeze(1)).any(dim=-1).float().mean().item()
            val_listener_top5_orig = (lst_top5 == y_val.unsqueeze(1)).any(dim=-1).float().mean().item()
            val_listener_top3_recon = (lst_top3 == yt_argmax.unsqueeze(1)).any(dim=-1).float().mean().item()
            val_listener_top5_recon = (lst_top5 == yt_argmax.unsqueeze(1)).any(dim=-1).float().mean().item()

            # Listener loss (soft CE between listener and y_tilde)
            log_p_lst = F.log_softmax(listener_logits, dim=-1)
            val_ll = -(val_y_tilde * log_p_lst).sum(dim=-1).mean().item()

            # Consistency agreement: fraction of speaker's l selections in AE top-2l by |c|.
            F_dim = c_val.shape[1]
            k_target = min(2 * args.l, F_dim)
            topk_idx = c_val.abs().topk(k_target, dim=1).indices
            top_mask = torch.zeros_like(c_val).scatter_(1, topk_idx, 1.0)
            val_consistency_agree = ((m_u * top_mask).sum(dim=1) / args.l).mean().item()

            val_unique_utterances = len({tuple(row.tolist()) for row in val_u})

        wandb.log({
            "val/reward": val_rewards.mean().item(),
            "val/faith_top1": val_faith_top1,
            "val/faith_top3": val_faith_top3,
            "val/faith_top5": val_faith_top5,
            "val/consistency_agree": val_consistency_agree,
            "val/listener_loss": val_ll,
            "val/listener_acc_recon": val_listener_acc_recon,
            "val/listener_top3_recon": val_listener_top3_recon,
            "val/listener_top5_recon": val_listener_top5_recon,
            "val/listener_acc_orig": val_listener_acc_orig,
            "val/listener_top3_orig": val_listener_top3_orig,
            "val/listener_top5_orig": val_listener_top5_orig,
            "val/unique_utterances": val_unique_utterances,
            "iteration": t,
        })
        logger.info(f"  val/reward: {val_rewards.mean().item():.4f}")
        logger.info(f"  val/faith_top1: {val_faith_top1:.4f}  "
                    f"top3: {val_faith_top3:.4f}  top5: {val_faith_top5:.4f}")
        logger.info(f"  val/consistency_agree: {val_consistency_agree:.4f}  "
                    f"(random = {k_target/c_val.shape[1]:.3f})")
        logger.info(f"  val/listener_loss: {val_ll:.4f}")
        logger.info(f"  val/listener_acc_recon: {val_listener_acc_recon:.4f}  "
                    f"top3: {val_listener_top3_recon:.4f}  top5: {val_listener_top5_recon:.4f}")
        logger.info(f"  val/listener_acc_orig:  {val_listener_acc_orig:.4f}  "
                    f"top3: {val_listener_top3_orig:.4f}  top5: {val_listener_top5_orig:.4f}")
        logger.info(f"  val/unique_utterances: {val_unique_utterances}  "
                    f"(N_val = {val_u.shape[0]})")

        if args.get("save_checkpoint", False):
            ckpt_subdir = args.get("ckpt_subdir", "speaker_listener_cub")
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
