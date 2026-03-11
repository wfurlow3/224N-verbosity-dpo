# Performs QLoRA SFT on Modal
import argparse
import json
import os
import random
from datetime import datetime, timezone
from importlib.metadata import version
import inspect

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

import modal


app = modal.App("sft-mistral-qlora")
data_volume = modal.Volume.from_name("verbosity-data", create_if_missing=True)  # Data volume
outputs_volume = modal.Volume.from_name("verbosity-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("verbosity-hf-cache", create_if_missing=True)
TRAIN_PATH = "/root/repo/data/sft/train.json"
VAL_PATH = "/root/repo/data/sft/val.json"
OUTPUT_DIR = "/root/repo/outputs/sft_on_instruct_v2"  # Write outputs to mounted volume
RESUME_CHECKPOINT = None
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SEED = 0
MAX_SEQ_LENGTH = 1024
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 5
EVAL_STEPS = 50

image = (
    modal.Image.debian_slim(python_version="3.11")  # Base image
    .pip_install(
        "torch==2.10.0",
        "transformers==5.2.0",
        "accelerate==1.12.0",
        "bitsandbytes==0.49.2",
        "peft==0.18.1",
        "datasets==4.5.0",
        "huggingface_hub==1.4.0",
        "trl==0.17.0",
    )
    .add_local_file(
        "scripts/train/sft_train_qlora.py",
        remote_path="/root/repo/scripts/train/sft_train_qlora.py",
    )
)


def format_example(example, eos_token):
    instruction = (example["instruction"] or "").strip()
    output = (example["output"] or "").strip()
    text = f"<s>[INST] {instruction} [/INST] {output}</s>"  # Instruction-tuned prompt format
    if eos_token and not text.endswith(eos_token):  # Ensure EOS is present
        text = text + eos_token
    return {"text": text}


def write_blueprint(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_training(args):

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output folder
    logs_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    set_seed(SEED)  # Seed torch/transformers.
    random.seed(SEED)

    train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")  # Load train json
    val_ds = load_dataset("json", data_files=VAL_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:  # Use EOS as pad when missing
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(lambda x: format_example(x, tokenizer.eos_token), remove_columns=train_ds.column_names)  # Build text field
    val_ds = val_ds.map(lambda x: format_example(x, tokenizer.eos_token), remove_columns=val_ds.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # QLoRA 4-bit loading.
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(  # Load quantized base model
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # Needed with gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(  # LoRA adapter config
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # Prefer bf16 when supported
    train_args = {
        "output_dir": OUTPUT_DIR,
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_steps": args.max_steps,
        "eval_strategy": "steps",
        "eval_steps": EVAL_STEPS,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "logging_steps": LOGGING_STEPS,
        "logging_dir": logs_dir,
        "report_to": "none",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "seed": SEED,
        "data_seed": SEED,
    }


    training_args = TrainingArguments(**train_args)  # HF Trainer args
    base_to_dict = training_args.to_dict

    def to_dict_with_push_to_hub_token():
        d = base_to_dict()
        d.setdefault("push_to_hub_token", None)
        return d

    training_args.to_dict = to_dict_with_push_to_hub_token

    collator = None
    if DataCollatorForCompletionOnlyLM is not None:  # Loss only on assistant response tokens
        collator = DataCollatorForCompletionOnlyLM(response_template="[/INST]", tokenizer=tokenizer)

    sft_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": lora_config,
        "dataset_text_field": "text",
        "data_collator": collator,
        "max_seq_length": MAX_SEQ_LENGTH,
        "packing": False,
    }
    sft_init_params = inspect.signature(SFTTrainer.__init__).parameters # Deal with annoying trl api issues
    if "processing_class" in sft_init_params:
        sft_args["processing_class"] = tokenizer
    elif "tokenizer" in sft_init_params:
        sft_args["tokenizer"] = tokenizer
    sft_args = {k: v for k, v in sft_args.items() if k in sft_init_params}

    trainer = SFTTrainer(**sft_args)  # Build TRL trainer

    train_result = trainer.train(  # Start training loop
        resume_from_checkpoint=RESUME_CHECKPOINT if args.resume_checkpoint else None
    )
    trainer.save_model(OUTPUT_DIR)  # Save LoRA adapter
    tokenizer.save_pretrained(OUTPUT_DIR)

    blueprint = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": MODEL_NAME,
        "seed": SEED,
        "dataset": {
            "train_path": TRAIN_PATH,
            "val_path": VAL_PATH,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        },
        "hyperparams": {
            "smoke_test": bool(args.smoke_test),
            "max_steps": args.max_steps,
            "eval_steps": EVAL_STEPS,
            "save_steps": args.save_steps,
            "max_seq_length": MAX_SEQ_LENGTH,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "quantization": "4bit-nf4-double-quant",
            "precision": "bf16" if use_bf16 else "fp16",
        },
        "libraries": {
            "torch": version("torch"),
            "transformers": version("transformers"),
            "datasets": version("datasets"),
            "peft": version("peft"),
            "trl": version("trl"),
            "accelerate": version("accelerate"),
            "bitsandbytes": version("bitsandbytes"),
        },
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_result.metrics.get("train_samples_per_second", 0.0)),
    }

    write_blueprint(os.path.join(OUTPUT_DIR, "run_blueprint.json"), blueprint)  # Save run metadata
    write_jsonl(os.path.join(logs_dir, "train_log_history.jsonl"), trainer.state.log_history)

    print(f"Saved QLoRA adapter + tokenizer + manifest to {OUTPUT_DIR}")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    volumes={
        "/root/repo/data": data_volume,
        "/root/repo/outputs": outputs_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_training_modal(
    smoke_test=True,
    max_steps=2000,
    save_steps=200,
    resume_checkpoint: bool = False,
):
    smoke_test = str(smoke_test).lower() in {"1", "true", "yes", "y"}  # Normalize CLI bool
    max_steps = int(max_steps)  # Normalize CLI int
    save_steps = int(save_steps)  

    if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:  # Map token env var
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    os.environ["HF_HOME"] = "/root/.cache/huggingface"  # Use mounted HF cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume-checkpoint", action="store_true")
    args = parser.parse_args([])  # Build args from function params
    args.smoke_test = smoke_test
    args.max_steps = max_steps
    args.save_steps = save_steps
    args.resume_checkpoint = resume_checkpoint
    
    
    run_training(args)  # Run training inside Modal container
    outputs_volume.commit()  # Persist output artifacts
    hf_cache_volume.commit()


@app.local_entrypoint()
def modal_main(
    smoke_test: bool = True,
    max_steps: int = 2000,
    save_steps: int = 200,
    resume_checkpoint: bool = False,
):
    run_training_modal.remote(  # Kick off remote Modal job
        smoke_test=smoke_test,
        max_steps=max_steps,
        save_steps=save_steps,
        resume_checkpoint=resume_checkpoint,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume-checkpoint", action="store_true")
    args = parser.parse_args()
    run_training(args)  # Local training entrypoint


if __name__ == "__main__":
    main()
