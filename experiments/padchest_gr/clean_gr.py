"""
Build PadChest-GR CSV with enriched concept labels via LLM negation extraction.

Combines:
1. Abnormal findings (labels from GR JSON, already labeled)
2. Normal findings (abnormal=false sentences → LLM extracts negated concepts)
3. Original diagnosis labels from processed_labels.csv

Output format matches processed_labels.csv: ImageID + concept columns in {-1, 0, +1}
  +1 = concept present (from abnormal findings or original labels)
  -1 = concept explicitly absent (from normal findings via LLM or original labels)
   0 = unknown (not mentioned)

Usage:
    uv run experiments/padchest_gr/clean_gr.py
    uv run experiments/padchest_gr/clean_gr.py --skip_llm
"""

import argparse
import json
import time
from collections import Counter

import pandas as pd

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()

ROOT = get_root_dir()
GR_JSON = f"{ROOT}/data/PadChest-GR/grounded_reports_20240819.json"
PROCESSED_LABELS = f"{ROOT}/data/padchest/processed_labels.csv"
CACHE_FILE = f"{ROOT}/data/PadChest-GR/gr_normal_sentence_concepts.json"
OUT_CSV = f"{ROOT}/data/PadChest-GR/processed_labels_gr.csv"


DIAGNOSES = {
    "normal", "copd_signs", "pneumonia", "heart_insufficiency",
    "pulmonary_edema", "pulmonary_fibrosis", "emphysema",
    "tuberculosis", "tuberculosis_sequelae", "lung_metastasis",
    "post_radiotherapy_changes", "atypical_pneumonia", "bone_metastasis",
    "pulmonary_hypertension", "respiratory_distress", "asbestosis_signs",
    "lymphangitis_carcinomatosa", "pulmonary_artery_hypertension",
    "pulmonary_venous_hypertension", "lepidic_adenocarcinoma",
}
SPECIAL = {"exclude", "unchanged", "suboptimal_study"}
EXCLUDE_FROM_LLM = DIAGNOSES | SPECIAL


def get_all_concept_names(processed_labels_path):
    df = pd.read_csv(processed_labels_path, nrows=0)
    return [c for c in df.columns if c != "ImageID"]


def extract_concepts_with_llm(sentences, concept_names, cache_path, skip_llm=False):
    """Use Claude API to extract negated concepts from abnormal=false sentences."""
    cache = {}
    try:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached sentence->concept mappings")
    except FileNotFoundError:
        pass

    uncached = [s for s in sentences if s not in cache]
    print(f"Uncached sentences: {len(uncached)}")

    if skip_llm or not uncached:
        return cache

    import anthropic
    client = anthropic.Anthropic()

    concept_list_full = "\n".join(f"  - {c}" for c in concept_names)
    batch_size = 20

    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        sentences_str = "\n".join(f'{j+1}. "{s}"' for j, s in enumerate(batch))

        prompt = f"""You are analyzing chest X-ray radiology report sentences. Each sentence below describes a NORMAL finding (no abnormality). Your task is to identify which medical concepts are being explicitly negated or declared normal.

For each sentence, output the concept names from the following list that are being negated/declared absent. If the sentence is too vague (e.g., "No significant findings"), output "GENERAL_NORMAL" (meaning all concepts are normal but none specifically mentioned).

Concept list:
{concept_list_full}

Sentences:
{sentences_str}

For each sentence, respond with ONLY the concept names (comma-separated) or "GENERAL_NORMAL". One line per sentence, numbered to match.
Format:
1. concept1, concept2
2. GENERAL_NORMAL
3. concept1"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = response.content[0].text.strip()

            for line in result_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(".", 1)
                if len(parts) != 2:
                    continue
                try:
                    idx = int(parts[0].strip()) - 1
                except ValueError:
                    continue
                if idx < 0 or idx >= len(batch):
                    continue

                concepts_str = parts[1].strip()
                if concepts_str.upper() == "GENERAL_NORMAL":
                    cache[batch[idx]] = ["GENERAL_NORMAL"]
                else:
                    concepts = [c.strip().lower().replace(" ", "_")
                               for c in concepts_str.split(",")]
                    valid = [c for c in concepts if c in set(concept_names)]
                    cache[batch[idx]] = valid if valid else ["GENERAL_NORMAL"]

        except Exception as e:
            print(f"  API error: {e}")
            time.sleep(5)
            continue

        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {i + len(batch)}/{len(uncached)} sentences")

        time.sleep(0.5)

    return cache


def parse_args():
    p = argparse.ArgumentParser(description="Build clean PadChest-GR CSV")
    p.add_argument("--skip_llm", action="store_true",
                   help="Skip LLM calls, use only cached results")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading PadChest-GR JSON ...")
    with open(GR_JSON) as f:
        gr_data = json.load(f)
    print(f"  {len(gr_data)} images")

    all_concept_names = get_all_concept_names(PROCESSED_LABELS)
    print(f"  {len(all_concept_names)} concepts from processed_labels.csv")

    processed_df = pd.read_csv(PROCESSED_LABELS).set_index("ImageID")

    # Collect abnormal=false sentences
    normal_sentences = set()
    for entry in gr_data:
        for f in entry["findings"]:
            if not f["abnormal"]:
                normal_sentences.add(f["sentence_en"])
    print(f"  {len(normal_sentences)} unique abnormal=false sentences")

    # LLM: only findings (exclude diagnoses and special)
    finding_names_only = [c for c in all_concept_names if c not in EXCLUDE_FROM_LLM]
    print(f"  {len(finding_names_only)} findings for LLM")

    print("\nExtracting concepts from normal sentences ...")
    sentence_to_concepts = extract_concepts_with_llm(
        list(normal_sentences), finding_names_only, CACHE_FILE, skip_llm=args.skip_llm,
    )

    # Build per-image concept vectors
    print("\nBuilding concept vectors ...")
    rows = []

    for entry in gr_data:
        image_id = entry["ImageID"]

        # Start from processed_labels values
        if image_id in processed_df.index:
            concepts = {c: int(processed_df.loc[image_id, c]) for c in all_concept_names}
        else:
            concepts = {c: 0 for c in all_concept_names}

        # Only upgrade: 0 -> +1 or 0 -> -1
        for f in entry["findings"]:
            if f["abnormal"]:
                for label in f.get("labels", []) or []:
                    label_norm = label.lower().replace(" ", "_")
                    if label_norm in concepts and concepts[label_norm] == 0:
                        concepts[label_norm] = 1
            else:
                sentence = f["sentence_en"]
                negated = sentence_to_concepts.get(sentence, [])
                if "GENERAL_NORMAL" in negated:
                    pass  # too vague, skip
                else:
                    skip = DIAGNOSES | {"GENERAL_NORMAL"}
                    for c in negated:
                        if c in skip:
                            continue
                        if c in concepts and concepts[c] == 0:
                            concepts[c] = -1

        rows.append({"ImageID": image_id, **concepts})

    df = pd.DataFrame(rows)
    findings = sorted(all_concept_names)
    out = df[["ImageID"] + findings].copy()

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    print(f"  {len(out)} images, {len(findings)} columns")

    # Stats
    z = out[findings].values
    n_pos = (z == 1).sum()
    n_neg = (z == -1).sum()
    n_unk = (z == 0).sum()
    total = z.size
    print(f"\n  +1: {n_pos} ({100*n_pos/total:.1f}%)")
    print(f"  -1: {n_neg} ({100*n_neg/total:.1f}%)")
    print(f"   0: {n_unk} ({100*n_unk/total:.1f}%)")


if __name__ == "__main__":
    main()
