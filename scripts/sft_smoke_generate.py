import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def render_prompt(instruction):
    return f"<s>[INST] {instruction.strip()} [/INST]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--adapter_dir", type=str, default="outputs/sft_mistral_qlora")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    prompts = [
        "Explain overfitting in machine learning in simple terms.",
        "Write a short email asking for an extension on a homework deadline.",
        "Give three tips for studying efficiently for an exam.",
        "What is the difference between supervised and unsupervised learning?",
        "Describe gradient descent in one paragraph.",
    ]

    for i, p in enumerate(prompts, start=1):
        prompt = render_prompt(p)
        encoded = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = out[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(completion, skip_special_tokens=True).strip()
        print(f"\n=== Prompt {i} ===")
        print(p)
        print("--- Output ---")
        print(text)


if __name__ == "__main__":
    main()
