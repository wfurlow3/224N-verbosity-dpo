"""
Build prompt pool by sampling from UltraFeedback and HelpSteer2 (no full-dataset iteration).
Uses random indices + dataset.select() so we only iterate over sampled subsets.
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

# used for deduplication
def normalize_prompt(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", (text or "").strip()).split()).lower()


"""Sample up to n_want rows from dataset via random indices; apply filter and dedup."""
def _sample_from_dataset(
    dataset,
    prompt_key: str,
    source_label: str,
    sub_source_key: str | None,
    n_want: int,
    seen_normalized: set[str],
    rng,
) -> list[dict]:
    n_total = len(dataset)
    if n_total == 0 or n_want <= 0:
        return []
    k = min(n_total, max(n_want * 5, n_want + 1000))     # oversample indices so we still get approximately n_want after filtering and dedup.
    indices = rng.sample(range(n_total), k)
    subset = dataset.select(indices)  # only iterate through subset, not the full dataset.
    rows: list[dict] = []
    for idx, example in zip(indices, subset):
        if len(rows) >= n_want:
            break
        prompt = (example.get(prompt_key) or "").strip()
        if not prompt or len(prompt) < 10:
            continue
        key = normalize_prompt(prompt)
        if key and key in seen_normalized:
            continue
        if key:
            seen_normalized.add(key)
        sub = example.get(sub_source_key) if sub_source_key else source_label
        rows.append({
            "id": f"__{len(rows)}",
            "source": source_label,
            "prompt": prompt,
            "sub_source": sub if sub is not None else source_label,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build prompt pool by sampling from HF datasets.")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["ultrafeedback", "helpsteer2"],
        help="Datasets to use.",
    )
    ap.add_argument(
        "--n_prompts",
        type=int,
        default=1_000,
        help="Max prompts to write.",
    )
    ap.add_argument(
        "--frac_ultrafeedback",
        type=float,
        default=0.5,
        help="Fraction of n_prompts to take from UltraFeedback (rest from HelpSteer2).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/prompts/prompt_pool.jsonl"),
        help="Output JSONL path.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = __import__("random").Random(args.seed)
    seen_normalized: set[str] = set()
    rows: list[dict] = []

    use_uf = "ultrafeedback" in args.datasets
    use_hs = "helpsteer2" in args.datasets
    n_uf = int(args.n_prompts * args.frac_ultrafeedback) if (use_uf and use_hs) else (args.n_prompts if use_uf else 0)
    n_hs = args.n_prompts - n_uf if use_hs else 0

    if use_uf and n_uf > 0:
        ds_uf = load_dataset("openbmb/UltraFeedback", split="train", trust_remote_code=True)
        batch = _sample_from_dataset(
            ds_uf,
            prompt_key="instruction",
            source_label="ultrafeedback",
            sub_source_key="source",
            n_want=n_uf,
            seen_normalized=seen_normalized,
            rng=rng,
        )
        rows.extend(batch)
        print(f"UltraFeedback: {len(batch)} prompts (target {n_uf})")

    if use_hs and n_hs > 0:
        ds_hs = load_dataset("nvidia/HelpSteer2", split="train", trust_remote_code=True)
        batch = _sample_from_dataset(
            ds_hs,
            prompt_key="prompt",
            source_label="helpsteer2",
            sub_source_key=None,
            n_want=n_hs,
            seen_normalized=seen_normalized,
            rng=rng,
        )
        rows.extend(batch)
        print(f"HelpSteer2: +{len(batch)} prompts (target {n_hs})")

    # shuffle and cap to n_prompts (in case we have extra after deduplication)
    if len(rows) > args.n_prompts:
        rng.shuffle(rows)
        rows = rows[: args.n_prompts]
    else:
        rng.shuffle(rows)

    for i, r in enumerate(rows):
        r["id"] = f"p{i:06d}"

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {out_path}")


if __name__ == "__main__":
    main()
