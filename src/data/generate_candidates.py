"""
Generate three candidate types per prompt: concise, verbose, too-short.
Uses teacher (moonshot-v1-8k) with style-specific system prompts.
Writes candidates.jsonl where each line is a candidate with prompt_id, candidate_id, variant, text, meta (temperature, top_p, variant, stats (tokens, chars)).
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.teacher.kimi_client import call_kimi

VARIANTS = {
    "concise": {
        "system": (
            "Be as short as possible without losing any useful information. Balance quality and length.\n\n"
            "Rules:\n"
            "- Give every fact, step, and requirement the user asked for. Don't cut corners on the actual answer.\n"
            "- No preambles, intros, or outros. Just give the answer.\n"
            "- Don't repeat the question.\n"
            "- Avoid hedging, and don't list five ways to do something if one way is enough.\n"
            "- For coding tasks, send only the code. No markdown boxes, no explanations unless required by the prompt.\n"
            "- If the prompt has specific rules, follow it exactly but keep the steps brief.\n"
            "- As soon as the question is answered correctly, stop."
        ),
        "max_tokens": 500, # this is pretty high because some of the prompts are complex; we rely on the prompt for conciseness.
        "temperature": 0.4,
    },

    "verbose": {
        "system": (
            "Write a correct and complete answer, but be intentionally verbose.\n"
            "Rules:\n"
            "- Include extra background/context.\n"
            "- Add transitional phrases and mild redundancy (without adding new facts).\n"
            "- If applicable, provide a longer step-by-step explanation.\n"
            "- Do NOT introduce errors or hallucinated details.\n"
        ),
        "max_tokens": 1000,
        "temperature": 0.7,
    },

    "too_short": {
        "system": (
            "Write an answer that is SHORT enough to be incomplete and inaccurate.\n"
            "Rules:\n"
            "- Omit key details.\n"
            "- Write exactly one sentence.\n"
            "- Do not refuse to answer."
        ),
        "max_tokens": 50,
        "temperature": 0.2,
    },
}


TOP_P = 0.9

def token_count(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(re.findall(r"\S+", text)) * 1.3)


def _generate_one(
    prompt_id: str,
    prompt_text: str,
    variant: str,
    tokenizer,
    retries: int,
    retry_delay: float,
    delay: float,
) -> tuple[dict | None, bool]:
    """Call API for one (prompt, variant). Returns (row, skipped). On failure after retries, returns (None, True)."""
    config = VARIANTS[variant]
    text = None
    for attempt in range(retries + 1):
        try:
            text = call_kimi(
                prompt_text,
                system_prompt=config["system"],
                temperature=config["temperature"],
                top_p=TOP_P,
                max_tokens=config["max_tokens"],
            )
            break
        except Exception as e:
            if attempt < retries:
                time.sleep(retry_delay)
            else:
                print(f"Error {prompt_id} {variant} (skipping): {e}", file=sys.stderr)
                return (None, True)
    if text is None:
        return (None, True)
    time.sleep(delay)
    candidate_id = f"{prompt_id}_{variant}_v1"
    meta = {
        "temperature": config["temperature"],
        "top_p": TOP_P,
        "max_tokens": config["max_tokens"],
        "variant": variant,
    }
    stats = {"tokens": token_count(text, tokenizer), "chars": len(text)}
    row = {
        "prompt_id": prompt_id,
        "candidate_id": candidate_id,
        "variant": variant,
        "text": text,
        "meta": {**meta, "stats": stats},
    }
    return (row, False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate concise/verbose/too_short candidates per prompt.")
    ap.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts/split_train.jsonl"),
        help="Input prompts JSONL (id, source, prompt, sub_source).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/candidates/candidates.jsonl"),
        help="Output candidates JSONL.",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help="Which variants to generate (concise, verbose, too_short)",
    )
    ap.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="# of prompts to generate.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay (s) after each API call (per worker) to avoid rate limits.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent API calls.",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per API call on failure. Failed calls will not bewritten.",
    )
    ap.add_argument(
        "--retry_delay",
        type=float,
        default=5.0,
        help="Seconds to wait before retrying a failed API call.",
    )
    ap.add_argument(
        "--tokenizer",  
        type=str,
        default=None,
        help="HuggingFace tokenizer name for exact token count (e.g. gpt2, meta-llama/Llama-2-7b-hf). If unset, use rough estimate.",
    )
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Prompts file not found: {prompts_path}", file=sys.stderr)
        print("Run build_prompt_pool.py then split_prompts.py first.", file=sys.stderr)
        sys.exit(1)

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"Warning: could not load tokenizer {args.tokenizer}: {e}. Using rough count.", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = len(prompts) * len(args.variants)
    tasks = [
        (line["id"], line["prompt"], variant)
        for line in prompts
        for variant in args.variants
    ]

    rows: list[dict] = []
    skipped = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _generate_one,
                prompt_id,
                prompt_text,
                variant,
                tokenizer,
                args.retries,
                args.retry_delay,
                args.delay,
            ): (prompt_id, variant)
            for prompt_id, prompt_text, variant in tasks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"Progress: {completed}/{total} completed", file=sys.stderr)
            row, was_skipped = future.result()
            if was_skipped:
                skipped += 1
            elif row is not None:
                rows.append(row)

    # Stable order: by prompt_id then variant
    rows.sort(key=lambda r: (r["prompt_id"], r["variant"]))

    with open(args.out, "w") as out_f:
        for r in rows:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} candidates to {args.out}", end="")
    if skipped:
        print(f" ({skipped} skipped due to API errors)", end="")
    print()


if __name__ == "__main__":
    main()
