"""
Merge processed_labels.csv with enriched processed_labels_gr.csv and analyze.

Replaces GR image rows in the full 39K dataset with LLM-enriched labels.
Saves to data/padchest_final/processed_labels_merged.csv.

Usage:
    uv run experiments/padchest_gr/merge_and_analyze.py
"""

import pandas as pd

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()

ROOT = get_root_dir()
PROCESSED_LABELS = f"{ROOT}/data/padchest/processed_labels.csv"
PROCESSED_LABELS_GR = f"{ROOT}/data/PadChest-GR/processed_labels_gr.csv"
OUT_MERGED = f"{ROOT}/data/padchest_final/processed_labels_merged.csv"

DIAGNOSES = [
    "normal", "copd_signs", "pneumonia", "heart_insufficiency",
    "pulmonary_edema", "pulmonary_fibrosis", "emphysema",
    "tuberculosis", "tuberculosis_sequelae", "lung_metastasis",
    "post_radiotherapy_changes", "atypical_pneumonia", "bone_metastasis",
    "pulmonary_hypertension", "respiratory_distress", "asbestosis_signs",
    "lymphangitis_carcinomatosa", "pulmonary_artery_hypertension",
    "pulmonary_venous_hypertension", "lepidic_adenocarcinoma",
]


def main():
    import os

    print("Loading datasets ...")
    df_full = pd.read_csv(PROCESSED_LABELS)
    df_gr = pd.read_csv(PROCESSED_LABELS_GR)
    print(f"  Full PadChest: {len(df_full)} images")
    print(f"  PadChest-GR:   {len(df_gr)} images")

    # Align GR columns to full
    df_gr = df_gr[df_full.columns]

    # Replace GR rows
    gr_ids = set(df_gr["ImageID"])
    df_non_gr = df_full[~df_full["ImageID"].isin(gr_ids)]
    df_merged = pd.concat([df_non_gr, df_gr], ignore_index=True)
    print(f"\n  Merged: {len(df_merged)} images ({len(df_non_gr)} non-GR + {len(df_gr)} GR)")

    os.makedirs(os.path.dirname(OUT_MERGED), exist_ok=True)
    df_merged.to_csv(OUT_MERGED, index=False)
    print(f"  Saved to {OUT_MERGED}")

    # Analyze
    concept_cols = [c for c in df_merged.columns if c != "ImageID"]

    print(f"\n{'='*60}")
    print("PER-DIAGNOSIS COUNTS (images with +1)")
    print(f"{'='*60}")
    for d in DIAGNOSES:
        if d in df_merged.columns:
            n = (df_merged[d] == 1).sum()
            print(f"  {d:<45} {n:>6}")

    normal_mask = df_merged["normal"] == 1
    df_abnormal = df_merged[~normal_mask]
    print(f"\n  Normal: {normal_mask.sum()}")
    print(f"  Non-normal: {len(df_abnormal)}")

    # Concept stats in non-normal
    print(f"\n{'='*60}")
    print("PER-CONCEPT (non-normal images)")
    print(f"{'='*60}")
    results = []
    for c in concept_cols:
        n_pos = (df_abnormal[c] == 1).sum()
        n_neg = (df_abnormal[c] == -1).sum()
        results.append({"concept": c, "n_pos": n_pos, "n_neg": n_neg})
    results.sort(key=lambda x: x["n_pos"], reverse=True)

    print(f"\nTop 50 by +1:")
    print(f"  {'Concept':<45} {'n+1':>6} {'n-1':>6}")
    for r in results[:50]:
        print(f"  {r['concept']:<45} {r['n_pos']:>6} {r['n_neg']:>6}")

    print(f"\nTHRESHOLD SUMMARY (n+1 in non-normal)")
    for thresh in [30, 50, 100, 200, 500]:
        n = sum(1 for r in results if r["n_pos"] >= thresh)
        print(f"  n+1 >= {thresh:>4}: {n:>4} concepts")


if __name__ == "__main__":
    main()
