# Runs AlpacaEval generation on Modal for given model
import os
import subprocess

import modal

max_instances = 5  
num_shards = 1

app = modal.App("verbosity-gen-alpacaeval")

# Output volums to save data and cache
data_volume = modal.Volume.from_name("verbosity-data", create_if_missing=True) 
hf_cache_volume = modal.Volume.from_name("verbosity-hf-cache", create_if_missing=True) 
outputs_volume = modal.Volume.from_name("verbosity-outputs", create_if_missing=True)  

image = (
    modal.Image.debian_slim(python_version="3.11")  # Base image
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
        "scripts/eval/gen_alpacaeval_outputs.py",
        remote_path="/root/repo/scripts/eval/gen_alpacaeval_outputs.py",
    )
)


def build_gen_cmd(
    max_instances,
    num_shards,
    shard_id,
):
    cmd = [  # Command for one shard
        "python",
        "scripts/eval/gen_alpacaeval_outputs.py",
        "--max_instances",
        str(max_instances),
        "--num_shards",
        str(num_shards),
        "--shard_id",
        str(shard_id),
    ]
    return cmd


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
def run_gen(
    max_instances=max_instances,
    num_shards=num_shards,
    shard_id=0,
):
    env = os.environ.copy()
    if "HF_TOKEN" in env and "HUGGINGFACE_HUB_TOKEN" not in env:  # Map token env var
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]
    env["HF_HOME"] = "/root/.cache/huggingface"  # Store HF cache in mounted volume

    cmd = build_gen_cmd(
        max_instances=max_instances,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    subprocess.run(cmd, cwd="/root/repo", env=env, check=True)  # Run generation script
    data_volume.commit()  
    hf_cache_volume.commit()  


@app.local_entrypoint()
def main(
    max_instances: int = max_instances,
    num_shards: int = num_shards,
):
    max_instances = int(max_instances)
    num_shards = int(num_shards)

    run_kwargs = {
        "max_instances": max_instances,
        "num_shards": num_shards,
    }

    if num_shards == 1:  # Single shard path
        run_gen.remote(shard_id=0, **run_kwargs)
        return

    calls = []
    for shard_id in range(num_shards):
        call = run_gen.spawn(shard_id=shard_id, **run_kwargs)  # Start shard job
        calls.append(call)

    for call in calls:
        call.get()  # Wait for shard to finish
