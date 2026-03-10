"""
validate/filter candidates (e.g. length, quality) before SFT/DPO.
toss truncated, refusals, verbose incorrect, and concise incorrect responses. 
use LLM judge to validate correctness.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REFUSAL_PATTERN = re.compile(
  r"\b(I cannot|I can't|I'm unable|I am unable|I'm sorry|I don't have|"
  r"As an AI|I'm not able|I am not able)\b",
  re.IGNORECASE,
)
JUDGE_MODEL = "gpt-4o-mini"


def is_truncated(row: dict) -> bool:
  """True if response hit the token limit or ends mid-sentence."""
  text = row["text"].strip()
  meta = row["meta"]
  max_tokens = meta["max_tokens"]
  tokens = meta["stats"]["tokens"]
  if max_tokens is not None and tokens is not None and tokens >= max_tokens:
      return True
  if len(text) > 15:
      last = text[-1]
      if last in (",", ";"):
          return True
      # this can't capture everything, but I think this is good enough?
      if text.endswith(" and") or text.endswith(" or") or text.endswith(" but"):
          return True
  return False


def is_refusal_heuristic(row: dict) -> bool:
  """True if response seems like a refusal."""
  text = row["text"].strip()
  return bool(REFUSAL_PATTERN.search(text) or text=="No.") # sometimes the too short model just says "No."


def _judge_one(item: tuple) -> tuple:
  """Single judge call for parallel execution. Returns (candidate_id, variant, correct, error_msg)."""
  row, prompt_text, retry_delay, delay_after = item
  candidate_id = row["candidate_id"]
  variant = row["variant"].strip().lower()
  try:
    correct = judge_correct(prompt_text, row["text"], delay_before_retry=retry_delay)
    if delay_after > 0:
      time.sleep(delay_after)
    return (candidate_id, variant, correct, None)
  except Exception as e:
    return (candidate_id, variant, False, str(e))


def judge_correct(prompt: str, response: str, delay_before_retry: float = 10.0, max_retries: int = 3) -> bool:
  """Use OpenAI gpt-4o-mini as LLM judge. Returns True if YES. Retries on 429."""
  from openai import OpenAI

  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
  client = OpenAI(api_key=api_key)

  system = (
    "You are a judge. Answer only YES or NO. "
    "Is the assistant's response correct and complete for the prompt? "
    "Ignore length or style; focus on correctness and completeness."
  )
  user = f"Prompt:\n{prompt}\n\nAssistant response:\n{response}"

  for attempt in range(max_retries):
    try:
      resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
          {"role": "system", "content": system},
          {"role": "user", "content": user},
        ],
        max_tokens=10,
      )
      out = (resp.choices[0].message.content or "").strip().upper()
      return out.startswith("YES") or out == "Y"
    except Exception as e:
      err_str = str(e)
      if attempt < max_retries - 1 and ("429" in err_str or "rate" in err_str.lower()):
        time.sleep(delay_before_retry)
        continue
      raise
  return False


def main() -> None:
  ap = argparse.ArgumentParser(description="Validate candidates: drop truncated + refusals, judge correctness (OpenAI gpt-4o-mini), write validated JSONL.")
  ap.add_argument("--candidates", type=Path, default=Path("data/candidates/candidates_train.jsonl"), help="Input JSONL (like candidates_train.jsonl).")
  ap.add_argument("--prompts", type=Path, default=Path("data/prompts/split_train.jsonl"), help="Prompts JSONL (id, prompt) for judge (like split_train.jsonl).")
  ap.add_argument("--out", type=Path, default=Path("data/candidates/candidates_train_validated.jsonl"), help="Output validated JSONL (like candidates_train_validated.jsonl).")
  ap.add_argument("--workers", type=int, default=10, help="Parallel judge API calls.")
  ap.add_argument("--delay", type=float, default=0.05, help="Seconds each worker waits after a judge call.")
  ap.add_argument("--retry_delay", type=float, default=10.0, help="Seconds to wait before retrying.")
  args = ap.parse_args()

  candidates_path = Path(args.candidates)
  if not candidates_path.exists():
    print(f"Candidates file not found: {candidates_path}", file=sys.stderr)
    sys.exit(1)
  prompts_path = Path(args.prompts)
  if not prompts_path.exists():
    print(f"Prompts file not found: {prompts_path}", file=sys.stderr)
    sys.exit(1)
  if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not set", file=sys.stderr)
    sys.exit(1)

  prompt_by_id: dict[str, str] = {}
  with open(prompts_path) as f:
    for line in f:
      line = line.strip()
      if not line:
          continue
      rec = json.loads(line)
      prompt_by_id[rec["id"]] = rec["prompt"]

  rows = []
  with open(candidates_path) as f:
    for line in f:
      line = line.strip()
      if not line:
          continue
      rows.append(json.loads(line))

  n_total = len(rows)
  dropped_truncated: dict[str, int] = {}
  dropped_refusal: dict[str, int] = {}
  dropped_concise_judge = 0
  dropped_verbose_judge = 0

  # in the first pass we do the truncation/refusal, and then set aside candidates that need to be judged for correctness
  kept_without_judge: list[dict] = []
  to_judge: list[tuple] = []  # (row, prompt_text, retry_delay)
  for row in rows:
    variant = row["variant"].strip().lower()
    if is_truncated(row):
      dropped_truncated[variant] = dropped_truncated.get(variant, 0) + 1
      continue
    if is_refusal_heuristic(row):
      dropped_refusal[variant] = dropped_refusal.get(variant, 0) + 1
      continue
    if variant in ("concise", "verbose"):
      prompt_text = prompt_by_id.get(row["prompt_id"]) or ""
      if prompt_text:
        to_judge.append((row, prompt_text, args.retry_delay, args.delay))
        continue
    kept_without_judge.append(row)
  trunc_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped_truncated.items()))
  refusal_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped_refusal.items()))
  print(f"Dropped truncated: {sum(dropped_truncated.values())} ({trunc_str})", file=sys.stderr)
  print(f"Dropped refusals:  {sum(dropped_refusal.values())} ({refusal_str})", file=sys.stderr)

  n_judge = len(to_judge)
  print(f"Judging {n_judge} candidates with {args.workers} workers...", file=sys.stderr, flush=True)

  # start judging in paralllel
  judge_results: dict[str, bool] = {}  # candidate_id -> correct
  judge_errors: list[tuple[str, str]] = []  # (candidate_id, error)
  done = 0
  with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(_judge_one, t): t[0] for t in to_judge} 
    for future in as_completed(futures):
      done += 1
      if done % 100 == 0 or done == n_judge:
        print(f"Progress: {done}/{n_judge} judged", file=sys.stderr, flush=True)
      try:
        candidate_id, variant, correct, err = future.result()
        judge_results[candidate_id] = correct
        if err:
          judge_errors.append((candidate_id, err))
        if not correct:
          if variant == "concise":
            dropped_concise_judge += 1
          elif variant == "verbose":
            dropped_verbose_judge += 1
      except Exception as e:
        # task failed; just add to dropped
        row = futures[future]
        candidate_id = row["candidate_id"]
        judge_results[candidate_id] = False
        judge_errors.append((candidate_id, str(e)))
        if row["variant"].strip().lower() == "concise":
          dropped_concise_judge += 1
        else:
          dropped_verbose_judge += 1

  for candidate_id, err in judge_errors[:20]:
    print(f"Judge error for {candidate_id}: {err}", file=sys.stderr)
  if len(judge_errors) > 20:
    print(f"... and {len(judge_errors) - 20} more judge errors", file=sys.stderr)

  kept = kept_without_judge + [row for (row, _, _, _) in to_judge if judge_results.get(row["candidate_id"])]

  print(
    f"Validated: {len(kept)} kept, {n_total - len(kept)} dropped "
    f"(truncated={sum(dropped_truncated.values())}, refusal={sum(dropped_refusal.values())}, "
    f"concise_judge={dropped_concise_judge}, verbose_judge={dropped_verbose_judge})",
    file=sys.stderr,
  )
  print("Concise kept: ", len([row for row in kept if row["variant"] == "concise"]))
  print("Verbose kept: ", len([row for row in kept if row["variant"] == "verbose"]))
  print("Too short kept: ", len([row for row in kept if row["variant"] == "too_short"]))

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w") as f:
    for r in kept:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
  print(f"Wrote {len(kept)} candidates to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
