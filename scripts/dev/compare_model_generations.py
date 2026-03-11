"""Compare base instruct vs base+LoRA generations on Modal."""

import os
from typing import Dict, List, Tuple

import modal

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "/root/repo/outputs/sft_mistral_instruct"
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
GPU = "T4"
DEFAULT_PROMPTS = [
    "Explain overfitting in machine learning in simple terms.",
    "Write a short email asking for an extension on a homework deadline.",
]

app = modal.App("verbosity-compare-generations")
outputs_volume = modal.Volume.from_name("verbosity-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("verbosity-hf-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "huggingface_hub",
)


def render_prompt(instruction: str) -> str:
    # Match scripts/eval/gen_alpacaeval_outputs.py prompt formatting exactly.
    return f"<s>[INST] {instruction.strip()} [/INST]"


def load_tokenizer(base_model: str, adapter_dir: str, hf_token: str):
    from transformers import AutoTokenizer

    tokenizer_source = adapter_dir if adapter_dir else base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        token=hf_token if hf_token else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, tokenizer_source


def load_base_model(base_model: str, hf_token: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,
        token=hf_token if hf_token else None,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, dtype, ("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_completion(
    model,
    tokenizer,
    rendered_prompt: str,
    generation_kwargs: Dict,
) -> Tuple[str, int]:
    import torch

    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)

    prompt_len = encoded["input_ids"].shape[1]
    completion_ids = generated[0, prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text, int(completion_ids.shape[0])


def print_debug_header(
    base_model: str,
    adapter_dir: str,
    tokenizer_source: str,
    tokenizer,
    adapter_attached: bool,
    model_b,
):
    print("=" * 100)
    print("Debug info")
    print("=" * 100)
    print(f"Base model path/id: {base_model}")
    print(f"Adapter path: {adapter_dir}")
    print(f"Tokenizer source: {tokenizer_source}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Tokenizer name_or_path: {tokenizer.name_or_path}")
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")
    print(f"Tokenizer chat_template present: {bool(getattr(tokenizer, 'chat_template', None))}")
    print("Prompt formatter used: literal Mistral instruct format '<s>[INST] ... [/INST]'")
    print(f"Adapter attached to Model B: {adapter_attached}")
    print(f"Model B class: {model_b.__class__.__name__}")
    if adapter_attached and hasattr(model_b, "peft_config"):
        print(f"Model B adapter names: {list(model_b.peft_config.keys())}")
    print("=" * 100)


def print_comparison(
    prompt_idx: int,
    prompt: str,
    output_a: str,
    output_a_tokens: int,
    output_b: str,
    output_b_tokens: int,
):
    print(f"\nPrompt {prompt_idx}")
    print("-" * 100)
    print(f"Input: {prompt}")
    print("-" * 100)
    print("Model A (base instruct)")
    print(output_a)
    print(f"[length] tokens={output_a_tokens} chars={len(output_a)}")
    print("-" * 100)
    print("Model B (base + adapter)")
    print(output_b)
    print(f"[length] tokens={output_b_tokens} chars={len(output_b)}")
    print("-" * 100)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    volumes={
        "/root/repo/outputs": outputs_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_compare(prompts: List[str], adapter_dir: str = ADAPTER_DIR):
    import torch
    from huggingface_hub import login
    from peft import PeftModel

    if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "Missing HF token in Modal runtime. Set Modal secret 'huggingface' with key HF_TOKEN."
        )
    login(token=hf_token, add_to_git_credential=False)

    tokenizer, tokenizer_source = load_tokenizer(BASE_MODEL, adapter_dir, hf_token)
    generation_kwargs = {
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not DO_SAMPLE:
        generation_kwargs["temperature"] = 0.0

    model_a, dtype, device_map = load_base_model(BASE_MODEL, hf_token)
    outputs_a: List[Tuple[str, int]] = []
    for prompt in prompts:
        rendered = render_prompt(prompt)
        outputs_a.append(
            generate_completion(
                model=model_a,
                tokenizer=tokenizer,
                rendered_prompt=rendered,
                generation_kwargs=generation_kwargs,
            )
        )
    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_b_base, _, _ = load_base_model(BASE_MODEL, hf_token)
    model_b = PeftModel.from_pretrained(
        model_b_base,
        adapter_dir,
        token=hf_token if hf_token else None,
    )
    model_b.eval()
    adapter_attached = isinstance(model_b, PeftModel) or hasattr(model_b, "peft_config")

    print_debug_header(
        base_model=BASE_MODEL,
        adapter_dir=adapter_dir,
        tokenizer_source=tokenizer_source,
        tokenizer=tokenizer,
        adapter_attached=adapter_attached,
        model_b=model_b,
    )
    print(f"torch_dtype: {dtype}")
    print(f"device_map: {device_map}")
    print(f"HF token detected: {bool(hf_token)}")
    print(f"generation_kwargs: {generation_kwargs}")
    print("=" * 100)

    for i, prompt in enumerate(prompts, start=1):
        rendered = render_prompt(prompt)
        output_b, output_b_tokens = generate_completion(
            model=model_b,
            tokenizer=tokenizer,
            rendered_prompt=rendered,
            generation_kwargs=generation_kwargs,
        )
        output_a, output_a_tokens = outputs_a[i - 1]
        print_comparison(
            prompt_idx=i,
            prompt=prompt,
            output_a=output_a,
            output_a_tokens=output_a_tokens,
            output_b=output_b,
            output_b_tokens=output_b_tokens,
        )

    hf_cache_volume.commit()
    outputs_volume.commit()


@app.local_entrypoint()
def main(adapter_dir: str = ADAPTER_DIR, prompts: str = ""):
    prompt_list = [p.strip() for p in prompts.split("|||") if p.strip()] if prompts else DEFAULT_PROMPTS
    run_compare.remote(prompts=prompt_list, adapter_dir=adapter_dir)
