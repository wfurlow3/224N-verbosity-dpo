# Generates AlpacaEval model outputs

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # model for AlpacaEval generation.
ADAPTER_DIR = "/root/repo/outputs/sft_on_instruct_v2" # If using SFT / DPO model
OUTPUT_PATH = "data/alpacaeval/sft_on_instruct_v2_outputs.json"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 512
LOG_EVERY = 5


def load_alpacaeval_eval_split():
    ds = load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
        split="eval",
        trust_remote_code=True, 
    )
    return [dict(x) for x in ds]


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_instances", type=int, default=None) 
    parser.add_argument("--num_shards", type=int, default=1)  # Total shard count
    parser.add_argument("--shard_id", type=int, default=0) 
    args = parser.parse_args()
    shard_tag = f"[shard {args.shard_id}/{args.num_shards}]"
    output_path = OUTPUT_PATH
    if args.num_shards > 1:  # Write per-shard output file
        stem, ext = os.path.splitext(OUTPUT_PATH)
        output_path = f"{stem}.shard_{args.shard_id}{ext}"

    prompts = load_alpacaeval_eval_split()
    if args.max_instances is not None:  # Optional truncation before sharding
        prompts = prompts[: args.max_instances]
    prompts = prompts[args.shard_id :: args.num_shards]  # Take this shard's slice
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts for shard {args.shard_id}/{args.num_shards}",)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer_source = ADAPTER_DIR if ADAPTER_DIR else MODEL_NAME
    print(f"Loading tokenizer: {tokenizer_source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)  # Load tokenizer
    if tokenizer.pad_token is None:  # Use EOS as pad when missing
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left" # For decoding only

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # fp16 on GPU
    device_map = "auto" if torch.cuda.is_available() else None  # Let HF place model on GPU

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if ADAPTER_DIR: 
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    print("Model loaded. Starting generation loop.")

    outputs = []
    processed = 0

    for batch_idx, batch in enumerate(batched(prompts, BATCH_SIZE), start=1):  # Iterate prompt batches
        instructions = [row["instruction"] for row in batch]
        if LOG_EVERY > 0:
            start_i = processed + 1
            end_i = min(processed + len(batch), total_prompts)
            print(
                f"Generating batch {batch_idx}: prompts {start_i}-{end_i}/{total_prompts}",
                flush=True,
            )

        rendered = [f"<s>[INST] {instruction.strip()} [/INST]" for instruction in instructions]  # Mistral instruct format

        encoded = tokenizer(  # Tokenize a full batch
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if torch.cuda.is_available():  # Move batch tensors to model device
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad(): # For inference
            generated = model.generate(
                **encoded,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()  # Prompt token lengths per row
        decoded = []
        for i, prompt_len in enumerate(prompt_lens):
            completion_ids = generated[i, int(prompt_len) :]  # Remove prompt tokens, keep completion
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            decoded.append(text)

        for instruction, output in zip(instructions, decoded):
            outputs.append(
                {
                    "instruction": instruction,
                    "output": output.strip(),
                    "generator": f"{MODEL_NAME}+{ADAPTER_DIR}" if ADAPTER_DIR else MODEL_NAME,
                }
            )
            processed += 1
            if LOG_EVERY > 0 and (processed % LOG_EVERY == 0 or processed == total_prompts):
                print(
                    f"{shard_tag} Progress: {processed}/{total_prompts} prompts generated",
                    flush=True,
                )

    with open(output_path, "w", encoding="utf-8") as f:  # Save JSON output
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(outputs)} rows -> {output_path}")


if __name__ == "__main__":
    main()
