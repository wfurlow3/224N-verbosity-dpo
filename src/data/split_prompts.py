"""
This function splits prompt_pool.jsonl into split_train.jsonl, split_val.jsonl, split_test.jsonl.
90% train, 5% val, 5% test.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Split prompt pool by prompt id.")
    ap.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts/prompt_pool.jsonl"),
        help="Input prompt pool JSONL.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/prompts"),
        help="Directory for split_train/val/test.jsonl.",
    )
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.90,
        help="Fraction for train.",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction for val (test gets the rest).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    args = ap.parse_args()

    rows = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n = len(rows)
    rng = __import__("random").Random(args.seed)
    rng.shuffle(rows)

    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("split_train", train_rows), ("split_val", val_rows), ("split_test", test_rows)]:
        path = args.out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} prompts to {path}")
    print(f"Split: train={n_train} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()
