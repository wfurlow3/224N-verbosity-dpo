
import json
import os
import random

from datasets import load_dataset

DATASET_NAME = "OpenAssistant/oasst1"
SPLIT = "train"
TARGET_SIZE = 50000
TRAIN_RATIO = 0.95
SEED = 42
TRAIN_OUT = "data/sft/train.json"
VAL_OUT = "data/sft/val.json"


def extract_oasst_pairs(dataset_name, split):
    ds = load_dataset(dataset_name, split=split)
    rows = [dict(x) for x in ds]
    total_rows = len(rows)

    by_id = {}
    for row in rows:
        msg_id = row.get("message_id") or row.get("id")
        if msg_id:
            by_id[msg_id] = row

    pairs = []

    # Track filtering stats
    assistant_rows = 0
    has_parent = 0
    parent_role_ok = 0
    single_turn = 0
    non_empty = 0
    english_ok = 0
    language_reject_samples = []

    for row in rows:
        role = str(row.get("role", "")).lower()
        if role != "assistant":
            continue
        assistant_rows += 1

        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in by_id: # Not root message
            continue
        has_parent += 1

        parent = by_id[parent_id]
        parent_role = str(parent.get("role", "")).lower()
        if parent_role not in {"prompter", "user", "human"}: # Parent role not in user/prompter/human
            continue
        parent_role_ok += 1
        if parent.get("parent_id"):
            continue
        single_turn += 1

        instruction = (parent.get("text") or "").strip()
        output = (row.get("text") or "").strip()
        if not instruction or not output: # Not empty instruction/output
            continue
        non_empty += 1

        lang = (row.get("lang") or parent.get("lang") or "").lower()
        if lang and lang != "en": # Language not English
            if len(language_reject_samples) < 50:
                language_reject_samples.append(
                    {
                        "row_lang": row.get("lang"),
                        "parent_lang": parent.get("lang"),
                        "instruction": instruction,
                        "output": output,
                    }
                )
            continue
        english_ok += 1

        pairs.append({"instruction": instruction, "output": output})

    print(f"Filter debug - total rows: {total_rows}")
    print(f"Filter debug - assistant rows: {assistant_rows}")
    print(f"Filter debug - with valid parent: {has_parent}")
    print(f"Filter debug - parent role in user/prompter/human: {parent_role_ok}")
    print(f"Filter debug - single-turn root user prompt: {single_turn}")
    print(f"Filter debug - non-empty instruction/output: {non_empty}")
    print(f"Filter debug - english-only kept: {english_ok}")
    print(f"Filter debug - final pairs: {len(pairs)}")
    print(f"Filter debug - showing {len(language_reject_samples)} language-rejected samples:")
    for i, sample in enumerate(language_reject_samples, start=1):
        print(
            f"[lang-reject {i}] row_lang={sample['row_lang']} parent_lang={sample['parent_lang']}",
        )
        print(f"  instruction: {sample['instruction'][:300]}")
        print(f"  output: {sample['output'][:300]}")

    return pairs


def write_json(path, data):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    pairs = extract_oasst_pairs(DATASET_NAME, SPLIT) # Extract OASST pairs with filtering

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    pairs = pairs[:TARGET_SIZE] # Shuffle and truncate to target size

    total_pairs = len(pairs)
    train_n = int(total_pairs * TRAIN_RATIO)
    train_data = pairs[:train_n]
    val_data = pairs[train_n:]

    write_json(TRAIN_OUT, train_data)
    write_json(VAL_OUT, val_data)

    print(f"Prepared {total_pairs} total pairs")
    print(f"Train: {len(train_data)} -> {TRAIN_OUT}")
    print(f"Val:   {len(val_data)} -> {VAL_OUT}")


if __name__ == "__main__":
    main()
