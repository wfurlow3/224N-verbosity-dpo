# Converts DPO jsonl splits to deduplicated SFT json splits (we don't need rejected responses for SFT)

import json
import os


def convert_split(input_paths, output_path, split_name):
    input_rows = 0
    output_rows = []
    seen_prompts = set()

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                input_rows += 1
                row = json.loads(line)
                prompt = row["prompt"]
                chosen = row["chosen"]

                if prompt in seen_prompts:
                    continue

                seen_prompts.add(prompt)
                output_rows.append({"instruction": prompt, "output": chosen})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)

    output_count = len(output_rows)
    duplicates_removed = input_rows - output_count
    print(f"Split: {split_name}")
    print(f"  Inputs: {', '.join(os.path.basename(p) for p in input_paths)}")
    print(f"  Input rows: {input_rows}")
    print(f"  Output rows: {output_count}")
    print(f"  Duplicates removed: {duplicates_removed}")


def main():
    os.makedirs("data/sft", exist_ok=True)

    # we put dpo train and val -> sft train, and then test -> sft val for larger train dataset
    convert_split(
        input_paths=["data/dpo/disentangled/dpo_train.jsonl", "data/dpo/disentangled/dpo_val.jsonl"],
        output_path="data/sft/train.json",
        split_name="sft_train",
    )
    convert_split(
        input_paths=["data/dpo/disentangled/dpo_test.jsonl"],
        output_path="data/sft/val.json",
        split_name="sft_val",
    )


if __name__ == "__main__":
    main()
