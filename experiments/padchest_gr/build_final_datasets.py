"""
Build final datasets from processed_labels_merged.csv.

Two modes:

--mode top4 (default, original behavior):
    Dataset 1 (autoencoder): padchest_autoencoder_final.csv
        - All non-normal + sampled normals (to match largest abnormal class)
        - Columns: ImageID + 78 concepts
    Dataset 2 (speaker-listener): padchest_clean_final.csv
        - Filtered to top 4 diagnoses
        - Columns: ImageID + 78 concepts + 4 diagnoses

    Concept selection: union of top-3-diagnostic concepts + top N frequent.

--mode full (new, all-classes/all-concepts):
    Single dataset: padchest_full.csv
        - All images (after dropping exclude/unchanged)
        - Columns: ImageID + 19 diagnoses (matching ViT head) + all concepts
    Concepts JSON: padchest_full_concepts.json

Outputs to data/padchest_final/.

Usage:
    uv run experiments/padchest_gr/build_final_datasets.py               # top4 mode
    uv run experiments/padchest_gr/build_final_datasets.py --mode full   # full mode
"""

import argparse
import json
import os

import pandas as pd

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()

ROOT = get_root_dir()
MERGED_CSV = f"{ROOT}/data/padchest_final/processed_labels_merged.csv"
GR_CSV = f"{ROOT}/data/PadChest-GR/processed_labels_gr.csv"
OUT_DIR = f"{ROOT}/data/padchest_final"
OUT_AE = f"{OUT_DIR}/padchest_autoencoder_final.csv"
OUT_SL = f"{OUT_DIR}/padchest_clean_final.csv"
OUT_CONCEPTS = f"{OUT_DIR}/padchest_concepts_final.json"

# Full mode outputs to a separate directory
OUT_DIR_FULL = f"{ROOT}/data/padchest_final_19class"
OUT_FULL = f"{OUT_DIR_FULL}/padchest_full.csv"
OUT_FULL_CONCEPTS = f"{OUT_DIR_FULL}/padchest_full_concepts.json"

# 19 diagnoses matching the ViT classifier head
DIAGNOSES_19 = [
    "normal", "copd_signs", "pneumonia", "heart_insufficiency",
    "pulmonary_edema", "pulmonary_fibrosis", "emphysema",
    "tuberculosis", "tuberculosis_sequelae", "lung_metastasis",
    "post_radiotherapy_changes", "atypical_pneumonia", "bone_metastasis",
    "pulmonary_hypertension", "respiratory_distress", "asbestosis_signs",
    "lymphangitis_carcinomatosa", "pulmonary_artery_hypertension",
    "lepidic_adenocarcinoma",
]

DIAGNOSES_ALL = {
    "copd_signs", "pneumonia", "heart_insufficiency",
    "pulmonary_edema", "pulmonary_fibrosis", "emphysema",
    "tuberculosis", "tuberculosis_sequelae", "lung_metastasis",
    "post_radiotherapy_changes", "atypical_pneumonia", "bone_metastasis",
    "pulmonary_hypertension", "respiratory_distress", "asbestosis_signs",
    "lymphangitis_carcinomatosa", "pulmonary_artery_hypertension",
    "pulmonary_venous_hypertension", "lepidic_adenocarcinoma",
}
SPECIAL = {"normal", "exclude", "unchanged", "suboptimal_study"}
EXCLUDE_FROM_FINDINGS = DIAGNOSES_ALL | SPECIAL

TOP_3_FOR_PRIOR = ["copd_signs", "pneumonia", "heart_insufficiency"]
SL_DIAGNOSES = ["normal", "copd_signs", "pneumonia", "heart_insufficiency"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["top4", "full"], default="top4",
                   help="top4 = original 78-concept/4-diagnosis datasets; "
                        "full = single workable-class/all-concepts dataset")
    p.add_argument("--z_thresh", type=int, default=3,
                   help="|z|>=z_thresh for concepts in top 3 diagnoses (top4 mode only)")
    p.add_argument("--top_freq_n", type=int, default=50,
                   help="Top N most frequent concepts to add (top4 mode only)")
    p.add_argument("--min_class_positives", type=int, default=20,
                   help="Full mode: drop diagnosis columns with fewer positives than this "
                        "(default 20 keeps ~14 workable classes out of the 19 ViT head outputs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_concepts(gr_df, merged_df, z_thresh, top_freq_n):
    """Union of top-3 diagnostic concepts + top frequent concepts."""
    all_findings = [c for c in gr_df.columns if c != "ImageID" and c not in EXCLUDE_FROM_FINDINGS]
    print(f"All candidate findings: {len(all_findings)}")

    union = set()
    for d in TOP_3_FOR_PRIOR:
        mask = gr_df[d] == 1
        sub = gr_df.loc[mask, all_findings]
        known = (sub != 0).sum()
        selected = known[known >= z_thresh].index.tolist()
        union |= set(selected)
        print(f"  {d}: {len(selected)} concepts with |z|>={z_thresh}")
    print(f"Union: {len(union)}")

    non_normal = merged_df[merged_df["normal"] != 1]
    pos_counts = (non_normal[all_findings] == 1).sum().sort_values(ascending=False)
    top_freq = set(pos_counts.head(top_freq_n).index)
    new_from_freq = top_freq - union
    union |= top_freq

    print(f"Top {top_freq_n} frequent: {len(top_freq)} ({len(new_from_freq)} new)")
    print(f"Final concepts: {len(union)}")
    return sorted(union)


def build_full(merged, seed, min_class_positives):
    """Full mode: workable-class / all-concepts dataset with normals subsampled
    to match the largest non-normal class (keeps concept-BCE calibration fair).
    """
    # Exclude diagnosis columns + exclude/unchanged/suboptimal_study from concept set
    diag_cols_all = set(DIAGNOSES_19) | {"pulmonary_venous_hypertension"}
    concept_blacklist = diag_cols_all | {"exclude", "unchanged", "suboptimal_study"}
    concept_cols = [
        c for c in merged.columns
        if c != "ImageID" and c not in concept_blacklist
    ]
    missing = [d for d in DIAGNOSES_19 if d not in merged.columns]
    if missing:
        raise ValueError(f"Diagnosis columns missing from merged CSV: {missing}")

    # Filter diagnosis columns by minimum positive count
    diag_counts_pre = {d: int((merged[d] == 1).sum()) for d in DIAGNOSES_19}
    kept_diagnoses = [d for d in DIAGNOSES_19 if diag_counts_pre[d] >= min_class_positives]
    dropped_diagnoses = [d for d in DIAGNOSES_19 if diag_counts_pre[d] < min_class_positives]
    print(f"\n  Keeping diagnoses with >={min_class_positives} positives ({len(kept_diagnoses)}/{len(DIAGNOSES_19)}):")
    print(f"    kept:    {kept_diagnoses}")
    print(f"    dropped: {dropped_diagnoses}")

    # Subsample normals to match the largest non-normal kept diagnosis class size
    normal_mask = merged["normal"] == 1
    df_normal = merged[normal_mask]
    df_non_normal = merged[~normal_mask]
    non_normal_counts = {
        d: int((df_non_normal[d] == 1).sum())
        for d in kept_diagnoses if d != "normal"
    }
    largest_nonnormal, largest_count = max(non_normal_counts.items(), key=lambda x: x[1])
    print(f"  Normals (before sampling): {len(df_normal)}")
    print(f"  Non-normal images:         {len(df_non_normal)}")
    print(f"  Largest non-normal class:  {largest_nonnormal} ({largest_count})")

    n_sample = min(largest_count, len(df_normal))
    df_normal_sampled = df_normal.sample(n=n_sample, random_state=seed)
    print(f"  Sampled normals:           {len(df_normal_sampled)}")

    combined = pd.concat([df_non_normal, df_normal_sampled], ignore_index=True)
    out = combined[["ImageID"] + kept_diagnoses + concept_cols].copy()

    os.makedirs(OUT_DIR_FULL, exist_ok=True)
    out.to_csv(OUT_FULL, index=False)
    with open(OUT_FULL_CONCEPTS, "w") as f:
        json.dump(concept_cols, f, indent=2)
    print(f"\nSaved:\n  {OUT_FULL}  ({len(out)} rows x {len(out.columns)} cols)")
    print(f"  {OUT_FULL_CONCEPTS}  (F = {len(concept_cols)})")

    print("\n" + "=" * 60)
    print(f"DIAGNOSIS STATS ({len(kept_diagnoses)} kept classes)")
    print("=" * 60)
    for d in kept_diagnoses:
        n_pos = int((out[d] == 1).sum())
        print(f"  {d:<32} {n_pos:>8}   val(20%) ~= {int(n_pos * 0.2)}")

    print("\n" + "=" * 60)
    print(f"CONCEPT STATS ({len(concept_cols)} concepts)")
    print("=" * 60)
    mat = out[concept_cols].values
    total = mat.size
    n_pos = (mat == 1).sum()
    n_neg = (mat == -1).sum()
    n_unk = (mat == 0).sum()
    print(f"  +1: {n_pos:>10} ({100 * n_pos / total:.2f}%)")
    print(f"  -1: {n_neg:>10} ({100 * n_neg / total:.2f}%)")
    print(f"   0: {n_unk:>10} ({100 * n_unk / total:.2f}%)")


def main():
    args = parse_args()

    print("Loading datasets ...")
    merged = pd.read_csv(MERGED_CSV)
    print(f"  Merged: {len(merged)} images")

    # Drop exclude and unchanged images
    before = len(merged)
    merged = merged[(merged.get("exclude", 0) != 1) & (merged.get("unchanged", 0) != 1)].reset_index(drop=True)
    print(f"  After dropping exclude/unchanged: {len(merged)} (dropped {before - len(merged)})")

    if args.mode == "full":
        build_full(merged, args.seed, args.min_class_positives)
        return

    # --- top4 mode (original) ---
    gr = pd.read_csv(GR_CSV)
    print(f"  GR: {len(gr)} images")

    # Select concepts
    print("\n-- Concept selection --")
    concepts = select_concepts(gr, merged, args.z_thresh, args.top_freq_n)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CONCEPTS, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"Saved concepts to {OUT_CONCEPTS}")

    # Sample normals to match largest abnormal class
    normal_mask = merged["normal"] == 1
    df_normal = merged[normal_mask]
    df_non_normal = merged[~normal_mask]
    print(f"\n  Normal: {len(df_normal)}")
    print(f"  Non-normal: {len(df_non_normal)}")

    abnormal_counts = {d: (df_non_normal[d] == 1).sum() for d in DIAGNOSES_ALL if d in df_non_normal.columns}
    largest_class, largest_count = max(abnormal_counts.items(), key=lambda x: x[1])
    print(f"  Largest abnormal class: {largest_class} ({largest_count})")

    n_sample = min(largest_count, len(df_normal))
    df_normal_sampled = df_normal.sample(n=n_sample, random_state=args.seed)
    print(f"  Sampled {len(df_normal_sampled)} normal images")

    # Dataset 1: autoencoder
    print("\n-- Dataset 1: autoencoder --")
    df_ae = pd.concat([df_non_normal, df_normal_sampled], ignore_index=True)
    out_ae = df_ae[["ImageID"] + concepts].copy()
    out_ae.to_csv(OUT_AE, index=False)
    print(f"  {len(out_ae)} images, {len(concepts)} concepts")
    print(f"  Saved to {OUT_AE}")

    # Dataset 2: speaker-listener
    print("\n-- Dataset 2: speaker-listener --")
    has_sl_diag = pd.Series(False, index=df_ae.index)
    for d in SL_DIAGNOSES:
        if d in df_ae.columns:
            has_sl_diag |= (df_ae[d] == 1)
    df_sl = df_ae[has_sl_diag].reset_index(drop=True)

    out_sl = df_sl[["ImageID"] + concepts + SL_DIAGNOSES].copy()
    out_sl.to_csv(OUT_SL, index=False)
    print(f"  {len(out_sl)} images, {len(concepts)} concepts + {len(SL_DIAGNOSES)} diagnoses")
    print(f"  Saved to {OUT_SL}")

    # Stats
    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)

    for name, df in [("Autoencoder", out_ae), ("Speaker-Listener", out_sl)]:
        print(f"\n{name}: {len(df)} images")
        mat = df[concepts].values
        total = mat.size
        n_pos = (mat == 1).sum()
        n_neg = (mat == -1).sum()
        n_unk = (mat == 0).sum()
        print(f"  +1: {n_pos:>8} ({100*n_pos/total:.2f}%)")
        print(f"  -1: {n_neg:>8} ({100*n_neg/total:.2f}%)")
        print(f"   0: {n_unk:>8} ({100*n_unk/total:.2f}%)")

    print(f"\nSpeaker-Listener diagnosis counts:")
    for d in SL_DIAGNOSES:
        n = (out_sl[d] == 1).sum()
        print(f"  {d:<25} {n:>6}")


if __name__ == "__main__":
    main()
