"""
build DPO datasets from validated candidates.
groups by prompt_id, then for each prompt creates (prompt, chosen, rejected) pairs:
  - chosen = concise, rejected = verbose (prefer concise when both correct)
  - chosen = concise, rejected = too_short (when both exist)
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build DPO JSONL pairs from validated candidates."
    )
    ap.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/candidates/candidates_train_validated.jsonl"),
        help="Validated candidates JSONL.",
    )
    ap.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts/split_train.jsonl"),
        help="Prompts JSONL (id, prompt) to get prompt text.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/dpo/dpo_train.jsonl"),
        help="Output DPO JSONL.",
    )
    args = ap.parse_args()

    candidates_path = Path(args.candidates)
    prompts_path = Path(args.prompts)
    out_path = Path(args.out)

    if not candidates_path.exists():
        print(f"Candidates file not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)
    if not prompts_path.exists():
        print(f"Prompts file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    prompt_by_id: dict[str, str] = {} # map prompt_id to the actual prompt
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt_by_id[rec["id"]] = rec["prompt"]

    # prompt_variant[prompt_id][variant]
    prompt_variant: dict[str, dict[str, dict]] = {}
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = row["prompt_id"]
            variant = row["variant"].strip().lower()
            if not prompt_id or not variant:
                continue
            if prompt_id not in prompt_variant:
                prompt_variant[prompt_id] = {}
            prompt_variant[prompt_id][variant] = row

    pairs: list[dict] = []
    for prompt_id, variants in prompt_variant.items():
        prompt_text = prompt_by_id.get(prompt_id)
        if not prompt_text:
            continue
        concise = variants.get("concise")
        verbose = variants.get("verbose")
        too_short = variants.get("too_short")

        if concise and verbose:
            pairs.append({
                "prompt": prompt_text,
                "chosen": concise["text"].strip(),
                "rejected": verbose["text"].strip(),
            })
        if concise and too_short:
            pairs.append({
                "prompt": prompt_text,
                "chosen": concise["text"].strip(),
                "rejected": too_short["text"].strip(),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} DPO pairs to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
