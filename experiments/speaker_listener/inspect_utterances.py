"""
Inspect speaker utterances on val images. Loads a saved SL checkpoint, picks a
stratified sample of val images, and prints the speaker's selected concepts
side by side with literal's top-l selection, the radiologist's z_gt concepts,
and the listener's prediction vs the true diagnosis.

Usage:
    uv run experiments/speaker_listener/inspect_utterances.py \
        --ckpt checkpoints/speaker_listener/sl_<run_name>_iter50.pt
    uv run experiments/speaker_listener/inspect_utterances.py \
        --ckpt ... --n-per-class 5 --output analysis/utterance_examples.txt
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as Fnn

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()
ROOT = Path(get_root_dir())

sys.path.insert(0, str(ROOT / "experiments" / "speaker_listener"))
from speaker import Speaker, LiteralSpeaker
from listener import Listener


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="SL checkpoint with speaker_state_dict, listener_state_dict, args, ...")
    p.add_argument("--n-per-class", type=int, default=5,
                   help="Number of val images per class to inspect (default: 5)")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to write output (also prints to stdout)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for which val images to pick")
    p.add_argument("--literal", action="store_true",
                   help="Use LiteralSpeaker (top-l by |c|) instead of the trained speaker. "
                        "When set, --ckpt is only used for the listener and the AE/args.")
    return p.parse_args()


def load_components(ckpt_path: Path, use_literal: bool):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]                      # dict (from OmegaConf.to_container)
    n_concepts = ckpt["n_concepts"]
    K = ckpt["K"]
    concept_names = ckpt["concept_names"]

    # AE encoder
    ae = torch.load(ROOT / args["autoencoder_ckpt"], map_location="cpu", weights_only=False)
    W_E = ae["encoder_weight"]
    encoder_bias = ae.get("encoder_bias", None)
    d = W_E.shape[1]

    # Embeddings (h) and labels
    dataset_dir = ROOT / "data" / args.get("dataset_dir", "padchest_final")
    emb = torch.load(dataset_dir / args["embeddings_file"], map_location="cpu", weights_only=False)
    image_ids = emb["ImageIDs"]
    h_all = emb["embeddings"]
    z_gt_all = emb["z_gt"]
    diag_names = emb["diagnosis_names"]
    y_hat_all = emb["y_hat"]                  # classifier on h hard predictions

    # Filter to SL images
    sl_df = pd.read_csv(dataset_dir / args["sl_dataset"]).set_index("ImageID")
    sl_ids = set(sl_df.index)
    keep = [i for i, iid in enumerate(image_ids) if iid in sl_ids]
    image_ids = [image_ids[i] for i in keep]
    h_all = h_all[keep]
    z_gt_all = z_gt_all[keep]
    y_hat_all = y_hat_all[keep]

    # Compute concept activations c = h @ W_E^T (+ bias)
    c_all = h_all @ W_E.T + (encoder_bias if encoder_bias is not None else 0.0)

    # Val split
    split = torch.load(dataset_dir / "split.pt", map_location="cpu", weights_only=False)
    val_ids_set = set(split["val_image_ids"])
    val_idx = [i for i, iid in enumerate(image_ids) if iid in val_ids_set]
    image_ids = [image_ids[i] for i in val_idx]
    h_all = h_all[val_idx]
    c_all = c_all[val_idx]
    z_gt_all = z_gt_all[val_idx]
    y_hat_all = y_hat_all[val_idx]

    # Speaker (literal or trained)
    if use_literal:
        speaker = LiteralSpeaker(n_concepts)
    else:
        speaker_input_dim = n_concepts if args["speaker_input"] == "concepts" else d
        speaker = Speaker(n_concepts, input_dim=speaker_input_dim)
        speaker.load_state_dict(ckpt["speaker_state_dict"])
    speaker.eval()

    # Listener — preserve the same logit_clamp the run used (not in state_dict).
    listener = Listener(
        n_concepts, K,
        logit_clamp=float(args.get("logit_clamp", 5.0)),
    )
    listener.load_state_dict(ckpt["listener_state_dict"], strict=False)
    listener.eval()

    # Speaker input source
    speaker_x = c_all if (use_literal or args["speaker_input"] == "concepts") else h_all

    return {
        "args": args, "concept_names": concept_names, "diag_names": diag_names,
        "image_ids": image_ids, "h": h_all, "c": c_all, "z_gt": z_gt_all,
        "y_hat": y_hat_all, "speaker_x": speaker_x,
        "speaker": speaker, "listener": listener,
        "sl_df": sl_df, "l": int(args["l"]), "K": K,
    }


def stratify_val(comps, n_per_class: int, seed: int):
    """Pick n_per_class val image indices per diagnosis, stratified by class.

    Uses true diagnosis from sl_df (single-label assumption: pick highest-prevalence class).
    """
    g = torch.Generator().manual_seed(seed)
    diag_names = comps["diag_names"]
    image_ids = comps["image_ids"]
    sl_df = comps["sl_df"]

    selected = []
    for k, name in enumerate(diag_names):
        # Indices where this class is +1 in the SL csv
        cands = [i for i, iid in enumerate(image_ids) if sl_df.loc[iid][name] == 1]
        if not cands:
            continue
        perm = torch.randperm(len(cands), generator=g)[:n_per_class].tolist()
        selected.extend([(cands[p], k) for p in perm])
    return selected


def format_concept(name: str, sign: int, c_val: float, max_name_len: int = 38) -> str:
    sym = "+" if sign > 0 else ("-" if sign < 0 else " ")
    return f"  {sym} {name:<{max_name_len}} (|c|={abs(c_val):.2f})"


def inspect(comps, n_per_class: int, seed: int) -> str:
    selected = stratify_val(comps, n_per_class, seed)
    speaker = comps["speaker"]
    listener = comps["listener"]
    speaker_x = comps["speaker_x"]
    c_all = comps["c"]
    z_gt = comps["z_gt"]
    y_hat = comps["y_hat"]
    image_ids = comps["image_ids"]
    concept_names = comps["concept_names"]
    diag_names = comps["diag_names"]
    sl_df = comps["sl_df"]
    l = comps["l"]

    out_lines = []
    for idx, true_k in selected:
        iid = image_ids[idx]
        true_class = diag_names[true_k]
        c_i = c_all[idx]                 # [F]
        x_i = speaker_x[idx:idx+1]       # [1, input_dim]
        z_i = z_gt[idx]                  # [F] in {-1, 0, +1}

        # Speaker's selection
        with torch.no_grad():
            indices, _ = speaker.sample(x_i, l)
            indices = indices[0]                          # [l]
            u = torch.zeros_like(c_i)
            u[indices] = c_i[indices].sign()              # ternary {-1, 0, +1}
            # Listener prediction
            listener_logits = listener(u.unsqueeze(0))    # [1, K] or [1, K] probs
            if listener.tau > 0:
                listener_probs = listener_logits[0]       # already sigmoid'd
            else:
                listener_probs = torch.sigmoid(listener_logits[0])
            listener_preds = (listener_probs > 0.5).float()

        # Literal's top-l (selected by |c|)
        literal_idx = c_i.abs().topk(l).indices

        # Ground-truth concepts (+1 in z_gt)
        gt_pos_idx = (z_i == 1).nonzero(as_tuple=True)[0].tolist()
        gt_neg_idx = (z_i == -1).nonzero(as_tuple=True)[0].tolist()

        # Format
        out_lines.append("=" * 88)
        out_lines.append(f"ImageID: {iid}   true: {true_class}")
        listener_str = ", ".join(
            f"{diag_names[k]}={'+'if listener_preds[k] else '-'}({listener_probs[k]:.2f})"
            for k in range(len(diag_names))
        )
        y_hat_str = ", ".join(
            f"{diag_names[k]}={'+' if y_hat[idx, k]==1 else '-'}"
            for k in range(len(diag_names))
        )
        out_lines.append(f"  classifier(h):  {y_hat_str}")
        out_lines.append(f"  listener(u):    {listener_str}")
        out_lines.append("")

        out_lines.append(f"Speaker's selection (l={l}):")
        # Sort by |c|
        sorted_indices = sorted(indices.tolist(), key=lambda j: -abs(c_i[j].item()))
        for j in sorted_indices:
            sign = int(c_i[j].sign().item())
            out_lines.append(format_concept(concept_names[j], sign, c_i[j].item()))

        out_lines.append("")
        out_lines.append(f"Literal's selection (top-{l} by |c|):")
        for j in literal_idx.tolist():
            sign = int(c_i[j].sign().item())
            out_lines.append(format_concept(concept_names[j], sign, c_i[j].item()))

        out_lines.append("")
        out_lines.append(f"Ground truth z_gt (radiologist):")
        if gt_pos_idx:
            for j in gt_pos_idx[:8]:
                out_lines.append(f"  + {concept_names[j]}")
        if gt_neg_idx:
            negs = ", ".join(concept_names[j] for j in gt_neg_idx[:5])
            extra = f" (+{len(gt_neg_idx) - 5} more)" if len(gt_neg_idx) > 5 else ""
            out_lines.append(f"  -negated: {negs}{extra}")
        if not gt_pos_idx and not gt_neg_idx:
            out_lines.append("  (no annotations)")

        # Overlap stats
        speaker_set = set(indices.tolist())
        literal_set = set(literal_idx.tolist())
        gt_pos_set = set(gt_pos_idx)
        out_lines.append("")
        out_lines.append(
            f"Overlap: speaker∩literal={len(speaker_set & literal_set)}/{l}, "
            f"speaker∩gt_pos={len(speaker_set & gt_pos_set)}/{len(gt_pos_set) if gt_pos_set else 0}, "
            f"literal∩gt_pos={len(literal_set & gt_pos_set)}/{len(gt_pos_set) if gt_pos_set else 0}"
        )
        out_lines.append("")

    return "\n".join(out_lines)


def main():
    args = parse_args()
    print(f"Loading checkpoint: {args.ckpt}")
    comps = load_components(args.ckpt, use_literal=args.literal)

    print(f"Speaker: {'literal' if args.literal else 'trained'}")
    print(f"Concepts (F): {len(comps['concept_names'])}")
    print(f"Diagnoses (K): {len(comps['diag_names'])}: {comps['diag_names']}")
    print(f"Val images: {len(comps['image_ids'])}")
    print()

    output = inspect(comps, n_per_class=args.n_per_class, seed=args.seed)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
