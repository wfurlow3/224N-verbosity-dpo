import json
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer


VANILLA_PATH = "data/dpo/vanilla/dpo_train.jsonl"
DISENTANGLED_PATH = "data/dpo/disentangled/dpo_train.jsonl"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def token_length(text):
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    vanilla_rows = load_jsonl(VANILLA_PATH)
    dis_rows = load_jsonl(DISENTANGLED_PATH)

    UF_chosen_lengths = [
        token_length(row.get("chosen", ""))
        for row in tqdm(vanilla_rows, desc="UF chosen tokenization")
    ]
    UF_rejected_lengths = [
        token_length(row.get("rejected", ""))
        for row in tqdm(vanilla_rows, desc="UF rejected tokenization")
    ]

    prompt_groups = defaultdict(list)
    for row in dis_rows:
        prompt_groups[row.get("prompt", "")].append(row)

    DIS_chosen_lengths = []
    DIS_rejected_verbose_lengths = []
    DIS_rejected_short_lengths = []
    valid_group_count = 0
    skipped_group_count = 0

    for rows in tqdm(prompt_groups.values(), desc="DIS tokenization"):
        if len(rows) != 2:
            skipped_group_count += 1
            continue

        valid_group_count += 1
        rejected_lengths = [token_length(rows[0].get("rejected", "")), token_length(rows[1].get("rejected", ""))]
        verbose_len = max(rejected_lengths)
        short_len = min(rejected_lengths)

        DIS_chosen_lengths.append(token_length(rows[0].get("chosen", "")))
        DIS_chosen_lengths.append(token_length(rows[1].get("chosen", "")))
        DIS_rejected_verbose_lengths.append(verbose_len)
        DIS_rejected_short_lengths.append(short_len)

    all_lengths = (
        UF_chosen_lengths
        + UF_rejected_lengths
        + DIS_chosen_lengths
        + DIS_rejected_verbose_lengths
        + DIS_rejected_short_lengths
    )
    xmax = max(all_lengths) if all_lengths else 1

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    print(f"UltraFeedback mean chosen length: {mean(UF_chosen_lengths):.2f}")
    print(f"UltraFeedback mean rejected length: {mean(UF_rejected_lengths):.2f}")
    print(f"Disentangled total prompt groups: {len(prompt_groups)}")
    print(f"Disentangled groups with exactly 2 rows: {valid_group_count}")
    print(f"Disentangled skipped groups: {skipped_group_count}")
    print(f"Disentangled mean chosen length: {mean(DIS_chosen_lengths):.2f}")
    print(f"Disentangled mean verbose rejected length: {mean(DIS_rejected_verbose_lengths):.2f}")
    print(f"Disentangled mean too-short rejected length: {mean(DIS_rejected_short_lengths):.2f}")
    plt.style.use("seaborn-v0_8-white")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    axes[0].hist(UF_chosen_lengths, bins=50, density=True, alpha=0.5, label="chosen", range=(0, xmax))
    axes[0].hist(UF_rejected_lengths, bins=50, density=True, alpha=0.5, label="rejected", range=(0, xmax))
    axes[0].set_title("UltraFeedback Preference Length Distribution")
    axes[0].set_xlabel("Response Length (tokens)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    plt.style.use("seaborn-v0_8-white")

    axes[0].set_xlim(0, 1500)
    axes[0].set_yscale("log")
    # axes[0].set_ylim(0, 100)

    axes[1].set_xlim(0, 1500)
    axes[1].set_yscale("log")
    axes[1].hist(DIS_chosen_lengths, bins=50, density=True, alpha=0.5, label="chosen (concise)", range=(0, xmax))
    axes[1].hist(
        DIS_rejected_verbose_lengths,
        bins=50,
        density=True,
        alpha=0.5,
        label="rejected (verbose)",
        range=(0, xmax),
    )
    axes[1].hist(
        DIS_rejected_short_lengths,
        bins=50,
        density=True,
        alpha=0.5,
        label="rejected (too short)",
        range=(0, xmax),
    )
    axes[1].set_title("Disentangled Dataset Length Distribution")
    axes[1].set_xlabel("Response Length (tokens)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
