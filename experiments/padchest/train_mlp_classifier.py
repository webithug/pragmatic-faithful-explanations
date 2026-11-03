"""
Train an MLP diagnosis head on frozen ConceptCLIP embeddings.

Input: embeddings/padchest_final/embeddings_conceptclip.pt
      (embeddings [N, 1152], diagnoses_gt [N, K] in {-1, 0, +1})

Binary rule for labels (same as ViT training):
    y = (diagnoses_gt == 1).float()   # {0, 1}
Drop rows where every diagnosis is 0 (treated as fully unknown) before training.

Train/val split read from data/padchest_final/cls_{train,val}.txt (stratified).

Writes the best-macro-AUC checkpoint to
    checkpoints/padchest_mlp/classifier_mlp.pt

Usage:
    uv run experiments/padchest/train_mlp_classifier.py
    uv run experiments/padchest/train_mlp_classifier.py --hidden 512 256 --epochs 200
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.train.monitor import Monitor, MonitorConfig

load_dotenv()
ROOT = Path(get_root_dir())


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description="MLP diagnosis head on ConceptCLIP embeddings")
    p.add_argument("--embeddings_file", default="embeddings_conceptclip.pt")
    p.add_argument("--dataset_dir", default="padchest_final")
    p.add_argument("--hidden", type=int, nargs="+", default=[256])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--pos_weight_max", type=float, default=50.0)
    p.add_argument("--ckpt_subdir", default="padchest_mlp")
    # wandb via Monitor — matches the ViT/AE training convention.
    p.add_argument("--wandb_project", default="padchest_cls")
    p.add_argument("--wandb_entity", default="wwang205-johns-hopkins-university")
    p.add_argument("--run_name", default=None)
    p.add_argument("--log_step", type=int, default=10)
    return p.parse_args()


def load_split(dataset_dir: Path) -> tuple[set, set]:
    with open(dataset_dir / "cls_train.txt") as f:
        train_ids = set(line.strip() for line in f)
    with open(dataset_dir / "cls_val.txt") as f:
        val_ids = set(line.strip() for line in f)
    return train_ids, val_ids


def build_arrays(data: dict, train_ids: set, val_ids: set):
    """Return X_tr, y_tr, X_va, y_va as tensors. y is binary (diag == 1)."""
    image_ids = data["ImageIDs"]
    emb = data["embeddings"]            # [N, D]
    diag = data["diagnoses_gt"]         # [N, K] in {-1, 0, +1}

    y = (diag == 1).float()             # binary
    # Drop rows where every column is 0 (fully unknown → no training signal).
    keep_mask = (diag != 0).any(dim=1)

    # Build index mask per split.
    train_idx, val_idx = [], []
    for i, iid in enumerate(image_ids):
        if not keep_mask[i]:
            continue
        if iid in train_ids:
            train_idx.append(i)
        elif iid in val_ids:
            val_idx.append(i)

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    return (
        emb[train_idx_t], y[train_idx_t],
        emb[val_idx_t], y[val_idx_t],
    )


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    logits = model(X.to(device)).cpu()
    probs = torch.sigmoid(logits).numpy()
    y_np = y.numpy()
    N, K = y_np.shape
    aucs, aps = [], []
    for k in range(K):
        n_pos = int(y_np[:, k].sum())
        n_neg = N - n_pos
        if n_pos > 0 and n_neg > 0:
            aucs.append(roc_auc_score(y_np[:, k], probs[:, k]))
            aps.append(average_precision_score(y_np[:, k], probs[:, k]))
        else:
            aucs.append(float("nan"))
            aps.append(float("nan"))
    macro_auc = float(np.nanmean(aucs))

    # Balanced accuracy @ threshold 0.5.
    preds = (probs > 0.5).astype(int)
    tprs, tnrs = [], []
    for k in range(K):
        gt = y_np[:, k]
        pr = preds[:, k]
        pos = gt == 1
        neg = gt == 0
        tprs.append(pr[pos].mean() if pos.any() else float("nan"))
        tnrs.append(1 - pr[neg].mean() if neg.any() else float("nan"))
    per_class_bal = 0.5 * (np.array(tprs) + np.array(tnrs))
    macro_bal = float(np.nanmean(per_class_bal))
    return {
        "macro_auc": macro_auc,
        "macro_bal_acc": macro_bal,
        "per_class_auc": aucs,
        "per_class_ap": aps,
        "per_class_bal": per_class_bal.tolist(),
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_dir = ROOT / "data" / args.dataset_dir
    emb_path = dataset_dir / args.embeddings_file
    print(f"Loading {emb_path}")
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    train_ids, val_ids = load_split(dataset_dir)
    X_tr, y_tr, X_va, y_va = build_arrays(data, train_ids, val_ids)
    diagnosis_names = data["diagnosis_names"]
    D = X_tr.shape[1]
    K = y_tr.shape[1]
    print(f"Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  D={D}  K={K}")
    print(f"MLP: {D} → {' → '.join(map(str, args.hidden))} → {K}  (dropout={args.dropout})")

    for k, name in enumerate(diagnosis_names):
        n_tr = int(y_tr[:, k].sum())
        n_va = int(y_va[:, k].sum())
        print(f"  {name:<28} train_pos={n_tr:>5}  val_pos={n_va:>5}")

    model = MLPClassifier(D, args.hidden, K, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Per-class pos_weight (clamped) on train set.
    n_pos = y_tr.sum(dim=0).clamp(min=1)
    n_neg = y_tr.shape[0] - n_pos
    pw = (n_neg / n_pos).clamp(max=args.pos_weight_max).to(device)
    print(f"pos_weight (clamped to {args.pos_weight_max}): "
          f"min={pw.min():.1f} median={pw.median():.1f} max={pw.max():.1f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    val_criterion = nn.BCEWithLogitsLoss()

    X_tr_dev = X_tr.to(device)
    y_tr_dev = y_tr.to(device)

    ckpt_dir = ROOT / "checkpoints" / args.ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "classifier_mlp.pt"

    run_name = args.run_name or f"train_mlp_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    monitor = Monitor(config=MonitorConfig(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        log_step=args.log_step,
        reduction="mean",
    ), run_config=vars(args))

    best_auc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(X_tr_dev.shape[0])
        epoch_loss, nb = 0.0, 0
        for i in range(0, X_tr_dev.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model(X_tr_dev[idx])
            loss = criterion(logits, y_tr_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            B = idx.shape[0]
            epoch_loss += loss.item()
            nb += 1
            monitor.update({"loss": loss.item() * B}, num_samples=B, prefix="train")

        if epoch % 5 == 0 or epoch == 1:
            tr_metrics = evaluate(model, X_tr, y_tr, device)
            va_metrics = evaluate(model, X_va, y_va, device)
            with torch.no_grad():
                va_loss = val_criterion(model(X_va.to(device)).cpu(), y_va).item()
            print(
                f"epoch {epoch:>3} | loss={epoch_loss/nb:.4f} "
                f"| val_loss={va_loss:.4f} "
                f"| train_auc={tr_metrics['macro_auc']:.4f} "
                f"| val_auc={va_metrics['macro_auc']:.4f} "
                f"| val_bal_acc={va_metrics['macro_bal_acc']:.4f}"
            )
            monitor.log({
                "loss": va_loss,
                "macro_auc": va_metrics["macro_auc"],
                "macro_bal_acc": va_metrics["macro_bal_acc"],
                "epoch": epoch,
            }, prefix="val")
            monitor.log({
                "macro_auc": tr_metrics["macro_auc"],
                "macro_bal_acc": tr_metrics["macro_bal_acc"],
                "epoch": epoch,
            }, prefix="train_eval")
            if va_metrics["macro_auc"] > best_auc:
                best_auc = va_metrics["macro_auc"]
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": D,
                    "hidden_dims": args.hidden,
                    "output_dim": K,
                    "dropout": args.dropout,
                    "diagnosis_names": diagnosis_names,
                    "epoch": epoch,
                    "val_macro_auc": best_auc,
                    "val_macro_bal_acc": va_metrics["macro_bal_acc"],
                }, ckpt_path)

    print(f"\nBest val macro_auc = {best_auc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")

    # Final per-class report on val.
    va_final = evaluate(model, X_va, y_va, device)
    print(f"\n{'diagnosis':<28} {'auc':>8} {'ap':>8} {'bal_acc':>8}")
    for k, name in enumerate(diagnosis_names):
        auc = va_final["per_class_auc"][k]
        ap = va_final["per_class_ap"][k]
        bal = va_final["per_class_bal"][k]
        print(f"{name:<28} {auc:>8.4f} {ap:>8.4f} {bal:>8.4f}")


if __name__ == "__main__":
    main()
