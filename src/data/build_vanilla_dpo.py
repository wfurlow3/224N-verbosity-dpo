"""
build vanilla DPO dataset from HuggingFaceH4/ultrafeedback_binarized.
this dataset already consolidated chosen/rejected pairs
Output: {"prompt": "...", "chosen": "...", "rejected": "..."} JSONL
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build vanilla DPO pairs from UltraFeedback binarized.")
    ap.add_argument("--n_train", type=int, default=8000, help="Number of train pairs.")
    ap.add_argument("--n_val", type=int, default=640, help="Number of val pairs.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--out_dir", type=Path, default=Path("data/dpo/vanilla"), help="Output directory.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HuggingFaceH4/ultrafeedback_binarized (train_prefs split)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    pairs = []
    skipped = 0
    for ex in ds:
        try:
            prompt = ex["prompt"]
            chosen = ex["chosen"][-1]["content"]
            rejected = ex["rejected"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            skipped += 1
            continue
        if not prompt or not chosen or not rejected:
            skipped += 1
            continue
        if chosen == rejected:
            skipped += 1
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(f"Extracted {len(pairs)} valid pairs ({skipped} skipped).")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_total = args.n_train + args.n_val
    if len(pairs) < n_total:
        print(f"Warning: only {len(pairs)} pairs available, wanted {n_total}.")
        n_total = len(pairs)

    train_pairs = pairs[:args.n_train]
    val_pairs = pairs[args.n_train:args.n_train + args.n_val]

    train_path = out_dir / "dpo_train.jsonl"
    val_path = out_dir / "dpo_val.jsonl"

    with open(train_path, "w") as f:
        for row in train_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_pairs)} train pairs to {train_path}")

    with open(val_path, "w") as f:
        for row in val_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(val_pairs)} val pairs to {val_path}")


if __name__ == "__main__":
    main()
