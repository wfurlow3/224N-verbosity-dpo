# Performs QLoRA DPO on Modal
import argparse
import json
import os
import random
from datetime import datetime, timezone
from importlib.metadata import version

import modal


app = modal.App("dpo-mistral-qlora")
data_volume = modal.Volume.from_name("verbosity-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name("verbosity-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("verbosity-hf-cache", create_if_missing=True)

DATA_PATH = "/root/repo/data/dpo/vanilla/dpo_train.jsonl"
VAL_PATH = "/root/repo/data/dpo/vanilla/dpo_val.jsonl"
OUTPUT_DIR = "/root/repo/outputs/simpo_on_instruct_vanilla"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SFT_ADAPTER_DIR =  None # Loads sft adapter if not training from instruct model
RESUME_CHECKPOINT = None # Allows resuming training from a checkpoint

# Training hyperparameters
SEED = 0
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-7
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 5
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3
BETA = 2.5
GAMMA = 0.25

image = ( # Define modal image
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.48,<5",
        "accelerate",
        "bitsandbytes",
        "peft",
        "datasets",
        "huggingface_hub",
        "trl==0.17.0",
    )
    .add_local_file(
        "scripts/train/simpo_train_qlora.py",
        remote_path="/root/repo/scripts/train/simpo_train_qlora.py",
    )
)


def ensure_eos(text, eos_token): # Ensures text ends with EOS token
    if not text:
        return text
    if eos_token and not text.endswith(eos_token):
        return text + eos_token
    return text


def format_example(example, eos_token): # Formats example for DPO training
    prompt = (example["prompt"] or "").strip()
    chosen = (example["chosen"] or "").strip()
    rejected = (example["rejected"] or "").strip()
    prompt_text = f"<s>[INST] {prompt} [/INST]"
    return {
        "prompt": prompt_text,
        "chosen": ensure_eos(chosen, eos_token),
        "rejected": ensure_eos(rejected, eos_token),
    }


def write_blueprint(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_model(
    torch,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PeftModel,
    prepare_model_for_kbit_training,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained( # Load quantized base model
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()


    if SFT_ADAPTER_DIR: # Load sft adapter if not training from instruct model
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR, is_trainable=True)
    return model


def run_training(args): # Runs SimPO training
    import torch # Include imports here to avoid local import issues
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        set_seed,
    )
    import trl
    import inspect
    DPOTrainer = trl.DPOTrainer
    DPOConfig = getattr(trl, "DPOConfig", None)

    class SimPOTrainer(DPOTrainer):
        def concatenated_forward(self, model, batch):
            output = super().concatenated_forward(model, batch)


            concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
            completion_mask = concatenated_batch["completion_attention_mask"]

            num_examples = batch["prompt_input_ids"].shape[0]
            output["chosen_lengths"] = completion_mask[:num_examples].sum(-1)
            output["rejected_lengths"] = completion_mask[num_examples:].sum(-1)

            return output

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            model_output = self.concatenated_forward(model, batch)

            policy_chosen_logps = model_output["chosen_logps"]
            policy_rejected_logps = model_output["rejected_logps"]
            policy_chosen_logits = model_output.get("mean_chosen_logits")
            policy_rejected_logits = model_output.get("mean_rejected_logits")

            chosen_lengths = model_output["chosen_lengths"].clamp_min(1).to(policy_chosen_logps.dtype)
            rejected_lengths = model_output["rejected_lengths"].clamp_min(1).to(policy_rejected_logps.dtype)

            chosen_avg_logp = policy_chosen_logps / chosen_lengths
            rejected_avg_logp = policy_rejected_logps / rejected_lengths
            logits = BETA * (chosen_avg_logp - rejected_avg_logp) - GAMMA
            loss = -F.logsigmoid(logits).mean()

            chosen_reward = BETA * chosen_avg_logp
            rejected_reward = BETA * rejected_avg_logp
            reward_margin = chosen_reward - rejected_reward
            reward_accuracy = (chosen_reward > rejected_reward).float().mean()

            prefix = "eval_" if train_eval == "eval" else ""
            metrics = {
                f"{prefix}loss": self.accelerator.gather_for_metrics(loss.detach()).mean().item(),
                f"{prefix}logps/chosen_avg": self.accelerator.gather_for_metrics(chosen_avg_logp.detach()).mean().item(),
                f"{prefix}logps/rejected_avg": self.accelerator.gather_for_metrics(rejected_avg_logp.detach()).mean().item(),
                f"{prefix}rewards/chosen": self.accelerator.gather_for_metrics(chosen_reward.detach()).mean().item(),
                f"{prefix}rewards/rejected": self.accelerator.gather_for_metrics(rejected_reward.detach()).mean().item(),
                f"{prefix}rewards/margins": self.accelerator.gather_for_metrics(reward_margin.detach()).mean().item(),
                f"{prefix}rewards/accuracies": self.accelerator.gather_for_metrics(reward_accuracy.detach()).mean().item(),
            }

            if policy_chosen_logits is not None:
                metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(
                    policy_chosen_logits.detach()
                ).mean().item()
            if policy_rejected_logits is not None:
                metrics[f"{prefix}logits/rejected"] = self.accelerator.gather_for_metrics(
                    policy_rejected_logits.detach()
                ).mean().item()

            return loss, metrics

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logs_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    set_seed(SEED)
    random.seed(SEED)

    train_ds = load_dataset("json", data_files=DATA_PATH, split="train")
    val_ds = load_dataset("json", data_files=VAL_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(lambda x: format_example(x, tokenizer.eos_token), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda x: format_example(x, tokenizer.eos_token), remove_columns=val_ds.column_names)

    policy_model = build_model(
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        BitsAndBytesConfig=BitsAndBytesConfig,
        PeftModel=PeftModel,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        "logging_steps": LOGGING_STEPS,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "logging_dir": logs_dir,
        "report_to": "none",
        "gradient_checkpointing": False,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "seed": SEED,
        "data_seed": SEED,
        "remove_unused_columns": False,
    }

    if DPOConfig is not None:
        training_args = DPOConfig(**train_args, beta=BETA)
    else:
        training_args = TrainingArguments(**train_args)
    dpo_init_params = inspect.signature(SimPOTrainer.__init__).parameters
    training_from_sft_adapter = SFT_ADAPTER_DIR is not None
    dpo_kwargs = {
        "model": policy_model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
    }
    if not training_from_sft_adapter:
        dpo_kwargs["peft_config"] = lora_config
    if "processing_class" in dpo_init_params:
        dpo_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in dpo_init_params:
        dpo_kwargs["tokenizer"] = tokenizer
    if "max_length" in dpo_init_params:
        dpo_kwargs["max_length"] = MAX_SEQ_LENGTH
    if "max_prompt_length" in dpo_init_params:
        dpo_kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH
    if "beta" in dpo_init_params:
        dpo_kwargs["beta"] = BETA

    # When starting from base model, ensure trainable LoRA params even if TRL doesn't accept peft_config.
    if (
        not training_from_sft_adapter
        and "peft_config" not in dpo_init_params
        and not isinstance(policy_model, PeftModel)
    ):
        policy_model = get_peft_model(policy_model, lora_config)
        dpo_kwargs["model"] = policy_model

    dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in dpo_init_params}

    trainer = SimPOTrainer(**dpo_kwargs)

    train_result = trainer.train(
        resume_from_checkpoint=RESUME_CHECKPOINT if args.resume_checkpoint else None
    )
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    blueprint = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": MODEL_NAME,
        "sft_adapter_dir": SFT_ADAPTER_DIR,
        "seed": SEED,
        "dataset": {
            "train_path": DATA_PATH,
            "train_size": len(train_ds),
            "val_path": VAL_PATH,
            "val_size": len(val_ds),
        },
        "hyperparams": {
            "smoke_test": bool(args.smoke_test),
            "max_steps": args.max_steps,
            "eval_steps": EVAL_STEPS,
            "save_steps": args.save_steps,
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "beta": BETA,
            "gamma": GAMMA,
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
        "train_samples_per_second": float(
            train_result.metrics.get("train_samples_per_second", 0.0)
        ),
    }

    write_blueprint(os.path.join(OUTPUT_DIR, "run_blueprint.json"), blueprint)
    write_jsonl(os.path.join(logs_dir, "train_log_history.jsonl"), trainer.state.log_history)

    print(f"Saved SimPO adapter + tokenizer + manifest to {OUTPUT_DIR}")


@app.function(
    image=image,
    gpu="H100",
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
    smoke_test = str(smoke_test).lower() in {"1", "true", "yes", "y"}
    max_steps = int(max_steps)
    save_steps = int(save_steps)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Allow CUDA to allocate more memory

    if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    os.environ["HF_HOME"] = "/root/.cache/huggingface"

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-checkpoint", action="store_true")
    args = parser.parse_args([])
    args.smoke_test = smoke_test
    args.max_steps = max_steps
    args.save_steps = save_steps
    args.resume_checkpoint = resume_checkpoint

    run_training(args)
    outputs_volume.commit()
    hf_cache_volume.commit()


@app.local_entrypoint()
def modal_main(
    smoke_test: bool = True,
    max_steps: int = 2000,
    save_steps: int = 200,
    resume_checkpoint: bool = False,
):
    run_training_modal.remote(
        smoke_test=smoke_test,
        max_steps=max_steps,
        save_steps=save_steps,
        resume_checkpoint=resume_checkpoint,
    )
