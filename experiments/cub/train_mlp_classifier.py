"""
Train an MLP classifier on frozen CLIP-ViT-L/14 embeddings for CUB.

Input: data/CUB/embeddings_clip.pt (built by extract_embeddings_clip.py).
Train/val split: read from `is_train` field in the embeddings file (the
official 80/20 CUB split).
Loss: cross-entropy over 200 classes.
Saves the best-top1 checkpoint to checkpoints/cub_mlp/classifier_mlp.pt.

Usage:
    uv run experiments/cub/train_mlp_classifier.py
    uv run experiments/cub/train_mlp_classifier.py --hidden 512 --epochs 200
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_file", default="embeddings_clip.pt")
    p.add_argument("--dataset_dir", default="CUB")
    p.add_argument("--hidden", type=int, nargs="+", default=[512])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ckpt_subdir", default="cub_mlp")
    p.add_argument("--wandb_project", default="cub_cls")
    p.add_argument("--wandb_entity", default="wwang205-johns-hopkins-university")
    p.add_argument("--run_name", default=None)
    p.add_argument("--log_step", type=int, default=10)
    return p.parse_args()


def build_arrays(data: dict):
    """Split embeddings into (train, val) by is_train. Class labels 0-indexed."""
    emb = data["embeddings"]
    cls = data["class_id"].long() - 1                      # 1-indexed -> 0-indexed
    is_train = data["is_train"].bool()
    return emb[is_train], cls[is_train], emb[~is_train], cls[~is_train]


@torch.no_grad()
def evaluate(model, X, y, device, K: int = 200):
    model.eval()
    logits = model(X.to(device)).cpu()
    loss = F.cross_entropy(logits, y).item()
    preds = logits.argmax(dim=-1)
    top1 = (preds == y).float().mean().item()
    _, top5 = logits.topk(5, dim=-1)
    top5_acc = (top5 == y.unsqueeze(1)).any(dim=-1).float().mean().item()

    # Macro balanced accuracy (per-class TPR + TNR / 2)
    y_np = y.numpy(); preds_np = preds.numpy()
    accs = []
    for k in range(K):
        pos = (y_np == k); neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        tpr = (preds_np[pos] == k).mean()
        tnr = (preds_np[neg] != k).mean()
        accs.append(0.5 * (tpr + tnr))
    macro_bal_acc = float(np.mean(accs)) if accs else float("nan")
    return {"loss": loss, "top1": top1, "top5": top5_acc,
            "macro_bal_acc": macro_bal_acc}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_dir = ROOT / "data" / args.dataset_dir
    emb_path = dataset_dir / args.embeddings_file
    print(f"Loading {emb_path}")
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    X_tr, y_tr, X_va, y_va = build_arrays(data)
    K = int(data["class_id"].max().item())
    D = X_tr.shape[1]
    class_names = data["class_names"]
    print(f"Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  D={D}  K={K}")
    print(f"MLP: {D} -> {' -> '.join(map(str, args.hidden))} -> {K}  "
          f"(dropout={args.dropout})")

    model = MLPClassifier(D, args.hidden, K, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
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

    best_top1 = 0.0
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
            epoch_loss += loss.item() * B
            nb += B
            monitor.update({"loss": loss.item() * B}, num_samples=B, prefix="train")

        if epoch % 5 == 0 or epoch == 1:
            tr = evaluate(model, X_tr, y_tr, device, K=K)
            va = evaluate(model, X_va, y_va, device, K=K)
            print(
                f"epoch {epoch:>3} | loss={epoch_loss/nb:.4f} "
                f"| val_loss={va['loss']:.4f} "
                f"| train_top1={tr['top1']:.4f} val_top1={va['top1']:.4f} "
                f"| val_top5={va['top5']:.4f}  bal_acc={va['macro_bal_acc']:.4f}"
            )
            monitor.log({
                "loss": va["loss"], "top1": va["top1"], "top5": va["top5"],
                "macro_bal_acc": va["macro_bal_acc"], "epoch": epoch,
            }, prefix="val")
            monitor.log({
                "top1": tr["top1"], "top5": tr["top5"],
                "macro_bal_acc": tr["macro_bal_acc"], "epoch": epoch,
            }, prefix="train_eval")
            if va["top1"] > best_top1:
                best_top1 = va["top1"]
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": D,
                    "hidden_dims": args.hidden,
                    "output_dim": K,
                    "dropout": args.dropout,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_top1": best_top1,
                    "val_top5": va["top5"],
                    "val_macro_bal_acc": va["macro_bal_acc"],
                }, ckpt_path)

    print(f"\nBest val top1 = {best_top1:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
