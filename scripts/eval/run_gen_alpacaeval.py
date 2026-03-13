# Generates AlpacaEval outputs using Modal
#
# script:
#   modal run scripts/eval/run_gen_alpacaeval.py \
#     --adapter_dir /root/repo/outputs/<model_dir> \
#     --output_path data/alpacaeval/<model_name>_outputs.json \
#     --generator_name <model_name>

import json
import os

import modal

app = modal.App("verbosity-gen-alpacaeval")
data_volume = modal.Volume.from_name("verbosity-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("verbosity-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("verbosity-outputs", create_if_missing=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 512
LOG_EVERY = 5

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets<3",
        "accelerate",
        "huggingface_hub",
        "alpaca-eval",
        "peft",
    )
    .add_local_file(
        "scripts/eval/run_gen_alpacaeval.py",
        remote_path="/root/repo/scripts/eval/run_gen_alpacaeval.py",
    )
)


## HELPERS
def load_alpacaeval_eval_split():
    from datasets import load_dataset
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


def run_generation(adapter_dir, output_path, generator_name, max_instances, num_shards, shard_id):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shard_tag = f"[shard {shard_id}/{num_shards}]"
    if num_shards > 1: # each shard needs its own output file
        stem, ext = os.path.splitext(output_path)
        output_path = f"{stem}.shard_{shard_id}{ext}"

    prompts = load_alpacaeval_eval_split()
    if max_instances is not None:
        prompts = prompts[:max_instances]
    prompts = prompts[shard_id::num_shards]
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts for shard {shard_id}/{num_shards}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer_source = adapter_dir if adapter_dir else MODEL_NAME
    print(f"Loading tokenizer: {tokenizer_source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # load base model, and if applicable, load fine-tuned adapater on top of base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("Model loaded. Starting generation loop.")

    outputs = []
    processed = 0

    for batch_idx, batch in enumerate(batched(prompts, BATCH_SIZE), start=1):
        instructions = [row["instruction"] for row in batch]
        if LOG_EVERY > 0:
            start_i = processed + 1
            end_i = min(processed + len(batch), total_prompts)
            print(f"Generating batch {batch_idx}: prompts {start_i}-{end_i}/{total_prompts}", flush=True)

        rendered = [f"<s>[INST] {instruction.strip()} [/INST]" for instruction in instructions]

        # tokenize,
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if torch.cuda.is_available():
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

        # generate response tokens, 
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()  
        # decode, 
        decoded = []
        for i, prompt_len in enumerate(prompt_lens):
            completion_ids = generated[i, int(prompt_len):]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            decoded.append(text)

        # and store outputs
        for instruction, output in zip(instructions, decoded):
            generator_label = generator_name or (f"{MODEL_NAME}+{adapter_dir}" if adapter_dir else MODEL_NAME)
            outputs.append({
                "instruction": instruction,
                "output": output.strip(),
                "generator": generator_label,
            })
            processed += 1
            if LOG_EVERY > 0 and (processed % LOG_EVERY == 0 or processed == total_prompts):
                print(f"{shard_tag} Progress: {processed}/{total_prompts} prompts generated", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(outputs)} rows -> {output_path}")


## MODAL LOGIC starts here
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    volumes={
        "/root/repo/data": data_volume,
        "/root/repo/outputs": outputs_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_gen_modal(
    max_instances=5,
    num_shards=1,
    shard_id=0,
    adapter_dir=None,
    output_path=None,
    generator_name=None,
):
    env = os.environ
    if "HF_TOKEN" in env and "HUGGINGFACE_HUB_TOKEN" not in env:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]
    os.environ["HF_HOME"] = "/root/.cache/huggingface"

    run_generation(
        adapter_dir=adapter_dir,
        output_path=output_path,
        generator_name=generator_name,
        max_instances=max_instances,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    data_volume.commit()
    hf_cache_volume.commit()


@app.local_entrypoint()
def main(
    max_instances: int = 5,
    num_shards: int = 1,
    adapter_dir: str = "",
    output_path: str = "",
    generator_name: str = "",
):
    adapter_dir = adapter_dir or None
    output_path = output_path or None
    generator_name = generator_name or None

    run_kwargs = {
        "max_instances": max_instances,
        "num_shards": num_shards,
        "adapter_dir": adapter_dir,
        "output_path": output_path,
        "generator_name": generator_name,
    }

    if num_shards == 1:
        run_gen_modal.remote(shard_id=0, **run_kwargs)
        return

    calls = []
    for shard_id in range(num_shards):
        call = run_gen_modal.spawn(shard_id=shard_id, **run_kwargs)
        calls.append(call)
    for call in calls:
        call.get()
