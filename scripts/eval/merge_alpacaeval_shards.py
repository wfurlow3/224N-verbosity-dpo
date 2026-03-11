# Merges sharded AlpacaEval outputs into a single JSON file

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_stem",
        type=str,
        default="data/alpacaeval/sft_on_instruct_v2/sft_on_instruct_v2_outputs.shard_*.json",
        help="Glob for shard JSON files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/alpacaeval/sft_on_instruct_v2/sft_on_instruct_v2_outputs.json",
        help="Path for merged JSON output.",
    )
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_stem))

    merged = []
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        merged.extend(rows)
        print(f"Loaded {len(rows)} rows from {path}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(shard_paths)} shards -> {len(merged)} rows")
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
