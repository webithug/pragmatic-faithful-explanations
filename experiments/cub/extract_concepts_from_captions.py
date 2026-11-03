"""
Extract per-caption 312-attribute mentions from CUB free-text descriptions.

For each of the 11,788 CUB images we have 10 human-written captions in
data/CUB_cvpr2016/text_c10/<class>/<image_basename>.txt. This script calls
Claude (haiku-4-5) with a prompt-cached attribute schema and asks it to label
each of the 10 captions with the subset of 312 attributes it mentions.

Output:
    data/CUB/cub_text_attribute_mentions.npy    int [11788, 312], counts in [0, 10]
        — how many of the 10 captions for image i mention attribute j
    data/CUB/cub_text_caption_extractions.json  cache
        — per-image list of 10 lists, each list is the attribute names extracted

Usage:
    uv run experiments/cub/extract_concepts_from_captions.py
    uv run experiments/cub/extract_concepts_from_captions.py --workers 16 --max-images 100   # smoke
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Pathp

import numpy as np
import pandas as pd

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()
ROOT = Path(get_root_dir())

CUB_DIR = ROOT / "data" / "CUB"
TEXT_DIR = ROOT / "data" / "CUB_cvpr2016" / "text_c10"
CACHE_PATH = CUB_DIR / "cub_text_caption_extractions.json"
OUT_NPY = CUB_DIR / "cub_text_attribute_mentions.npy"

MODEL = "gpt-4.1-mini"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16,
                   help="Number of concurrent API calls (default 16).")
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit to first N images (for smoke testing).")
    p.add_argument("--save-every", type=int, default=200,
                   help="Save cache every N processed images (default 200).")
    return p.parse_args()


def load_attribute_names() -> list[str]:
    path = CUB_DIR / "cub_concepts.json"
    with open(path) as f:
        return json.load(f)


def load_image_ids() -> list[str]:
    df = pd.read_csv(CUB_DIR / "cub_full.csv", usecols=["ImageID"])
    return df["ImageID"].tolist()


def captions_for(image_id: str) -> list[str]:
    """Read the 10 captions for a CUB image. Returns a list of <=10 strings."""
    # ImageID format: "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    class_dir, fname = image_id.split("/", 1)
    base = fname.rsplit(".", 1)[0]
    txt = TEXT_DIR / class_dir / f"{base}.txt"
    if not txt.exists():
        return []
    with open(txt) as f:
        return [ln.strip() for ln in f if ln.strip()]


SYSTEM_PROMPT_TEMPLATE = """You are an expert ornithologist annotating bird descriptions with a fixed schema of 312 attributes drawn from CUB-200-2011.

Each attribute is in the format `feature_group::value`, where feature_group describes a body part or whole-bird property and value is the specific observation. Examples:
  - has_bill_shape::dagger
  - has_wing_color::black
  - has_underparts_color::white
  - has_size::medium_(9_-_16_in)
  - has_belly_pattern::solid

The complete schema (use these EXACT strings — do not invent new ones):
{attribute_list}

# Task

You will receive 10 free-text bird descriptions for the same image. For EACH description independently, return the list of schema attributes that the description explicitly mentions or strongly implies.

# Detailed mapping rules

Color words: map "black", "white", "brown", "grey/gray", "red", "yellow", "blue", "olive", "rufous", "buff", "iridescent", "multi-colored", "orange", "pink", "green", "purple" to the matching `has_*_color::COLOR` value when paired with a body part. If the caption says "the bird is black", map to `has_primary_color::black`. If it names a part ("black bill", "white belly"), map to `has_bill_color::black` / `has_belly_color::white` etc. Do NOT map to multiple body parts unless the caption clearly says so.

Bill: "thin/narrow/needle-like" → has_bill_shape::needle. "thick/heavy/cone" → has_bill_shape::cone. "curved/hooked" → has_bill_shape::curved_(up_or_down) or has_bill_shape::hooked depending on which is more apt. "long" → has_bill_length::longer_than_head, "short" → has_bill_length::shorter_than_head, "about head length / medium" → has_bill_length::about_the_same_as_head.

Size: "small/tiny" → has_size::small_(5_-_9_in) or has_size::very_small_(3_-_5_in). "medium" → has_size::medium_(9_-_16_in). "large" → has_size::large_(16_-_32_in). "very large/huge" → has_size::very_large_(32_-_72_in).

Wings: "rounded/round" → has_wing_shape::rounded-wings. "pointed" → has_wing_shape::pointed-wings. "long-pointed" → has_wing_shape::long-pointed-wings. "tapered" → has_wing_shape::tapered-wings. "broad" → has_wing_shape::broad-wings.

Tail: "rounded" → has_tail_shape::rounded_tail. "pointed" → has_tail_shape::pointed_tail. "forked" → has_tail_shape::forked_tail. "fan-shaped" → has_tail_shape::fan-shaped_tail. "notched" → has_tail_shape::notched_tail. "squared" → has_tail_shape::squared_tail. Length: "long tail" → has_tail_pattern::long-tail (NOTE: tail length isn't in the schema as a separate attribute; encode shape only).

Patterns: "spotted" → has_*_pattern::spotted. "striped/streaked" → has_*_pattern::striped. "solid/uniform" → has_*_pattern::solid. "multi-colored" → has_*_pattern::multi-colored.

Crest: "crested/with crest" → has_crown_color matching the crest color, plus has_head_pattern::crested. Eyebrow stripes → has_head_pattern::eyebrow.

# Guard rails

- Do NOT invent attributes outside the schema. Map ONLY to the exact strings listed above.
- Do NOT include attributes the description does not support, even if you think they are typical for the species (you do not know the species).
- If a body part is not described, do not produce attributes for it.
- If a part is described with a color but the schema has no matching `has_<part>_color::<color>` entry, skip it rather than guess.
- "the bird is X colored" applies to has_primary_color::X — pick the dominant body color, not a body part.
- If a sentence mentions multiple body parts with different colors, emit one attribute per (part, color).
- Empty lists are fine. Most captions will produce 3-8 attributes. Captions emphasizing only one feature may produce 1-2.

# Output format

Return ONLY a JSON object, no commentary, no markdown fences:

{{"results": [["attr1", "attr2"], ["attr1"], ..., ["attr1", "attr2", "attr3"]]}}

The outer list has EXACTLY one entry per input caption, in the same order. Each inner list is the matched attribute names (use the EXACT strings from the schema)."""


def build_messages(system_text: str, captions: list[str]) -> list[dict]:
    user = "Captions:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
    user += '\n\nReturn STRICT JSON: {"results": [[...], [...], ...]} with one inner list per caption.'
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user},
    ]


def parse_response(text: str, n_captions: int, valid_attr_set: set[str]) -> list[list[str]]:
    """Parse Claude's JSON response. Returns list[list[str]] of length n_captions."""
    text = text.strip()
    # Strip ```json fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    results = obj.get("results", [])
    if len(results) != n_captions:
        raise ValueError(f"got {len(results)} results, expected {n_captions}")
    cleaned = []
    for row in results:
        if not isinstance(row, list):
            cleaned.append([])
            continue
        cleaned.append([a for a in row if isinstance(a, str) and a in valid_attr_set])
    return cleaned


def make_client():
    import openai
    return openai.OpenAI()


def extract_one(client, system_text, image_id, captions, valid_attr_set, retries=3):
    msgs = build_messages(system_text, captions)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=msgs,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            return parse_response(text, len(captions), valid_attr_set), resp.usage
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt * 2)


def main():
    args = parse_args()
    image_ids = load_image_ids()
    if args.max_images:
        image_ids = image_ids[: args.max_images]

    attr_names = load_attribute_names()
    valid_attr_set = set(attr_names)
    F = len(attr_names)
    N = len(image_ids)
    print(f"Processing {N} images, {F} attributes, model={MODEL}")

    # Load cache
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"  cache: {len(cache)} images already done")
    else:
        cache = {}

    # System prompt; OpenAI auto-caches prompt prefixes >1024 tokens.
    attribute_list_str = "\n".join(f"  - {n}" for n in attr_names)
    system_text = SYSTEM_PROMPT_TEMPLATE.format(attribute_list=attribute_list_str)

    client = make_client()
    todo = [iid for iid in image_ids if iid not in cache]
    print(f"  to process: {len(todo)}")
    if not todo:
        print("Nothing to do. Building output array from cache.")
    else:
        cache_lock = threading.Lock()
        save_lock = threading.Lock()
        cumul_in = cumul_out = cumul_cache_read = 0
        usage_lock = threading.Lock()
        done = 0
        t0 = time.time()

        def process(iid):
            nonlocal done
            captions = captions_for(iid)
            if not captions:
                with cache_lock:
                    cache[iid] = [[] for _ in range(10)]
                return
            results, usage = extract_one(client, system_text, iid, captions, valid_attr_set)
            # Pad/truncate to 10 (CUB always has exactly 10)
            results = results[:10] + [[] for _ in range(10 - len(results))]
            with cache_lock:
                cache[iid] = results
            with usage_lock:
                nonlocal cumul_in, cumul_out, cumul_cache_read
                cumul_in += getattr(usage, "prompt_tokens", 0) or 0
                cumul_out += getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "prompt_tokens_details", None)
                if details is not None:
                    cumul_cache_read += getattr(details, "cached_tokens", 0) or 0
            return iid

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, iid): iid for iid in todo}
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  FAILED {iid}: {e}")
                    continue
                done += 1
                if done % args.save_every == 0:
                    with cache_lock:
                        snapshot = dict(cache)
                    with save_lock:
                        with open(CACHE_PATH, "w") as f:
                            json.dump(snapshot, f)
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(todo)}  ({rate:.1f} img/s)  "
                          f"in={cumul_in}  out={cumul_out}  cache_read={cumul_cache_read}  "
                          f"ETA={eta/60:.1f} min")

        with cache_lock:
            snapshot = dict(cache)
        with open(CACHE_PATH, "w") as f:
            json.dump(snapshot, f)
        print(f"\nFinal usage: input={cumul_in}, output={cumul_out}, "
              f"cache_read={cumul_cache_read}")

    # Build [N, F] count tensor
    print("\nBuilding [N, F] count tensor ...")
    attr_to_idx = {a: j for j, a in enumerate(attr_names)}
    counts = np.zeros((N, F), dtype=np.int32)
    for i, iid in enumerate(image_ids):
        per_caption = cache.get(iid, [])
        seen = np.zeros(F, dtype=np.int32)
        for caption_attrs in per_caption[:10]:
            picked = {attr_to_idx[a] for a in caption_attrs if a in attr_to_idx}
            for j in picked:
                seen[j] += 1
        counts[i] = seen
    np.save(OUT_NPY, counts)
    print(f"Saved {OUT_NPY}  shape={counts.shape}  dtype={counts.dtype}")
    print(f"  total mentions: {counts.sum()}, mean per image: {counts.sum(axis=1).mean():.2f}")
    print(f"  attrs with >0 mentions: {(counts.sum(axis=0) > 0).sum()}/{F}")


if __name__ == "__main__":
    main()
