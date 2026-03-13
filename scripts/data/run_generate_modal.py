"""
Run generate_candidates.py on Modal, parallelized across N chunks.
"""
import json
import os
import subprocess
import sys

import modal

APP_NAME = "verbosity-dpo-generate"
VOLUME_NAME = "verbosity-dpo-candidates"
SECRET_NAME = "moonshot-api-key"

WORKERS = 10 # threads per container
DELAY = 0.1 # seconds between API calls per worker
RETRIES = 2
RETRY_DELAY = 5.0
CHECKPOINT_EVERY = 1000
TIMEOUT = 3600 * 5 # 5 hours per container
CONTAINERS = 5


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/workspace")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={"/out": volume},
    timeout=TIMEOUT,
)
def run_chunk(prompts_data: list[dict], out_filename: str):
    """Generate candidates for a list of prompts (passed as dicts). Writes to volume."""
    import tempfile

    # Write prompts to a temp file inside the container
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for row in prompts_data:
            f.write(json.dumps(row) + "\n")
        tmp_prompts = f.name

    out_path = f"/out/{out_filename}"
    cmd = [
        sys.executable, "-m", "src.data.generate_candidates",
        "--prompts", tmp_prompts,
        "--out", out_path,
        "--workers", str(WORKERS),
        "--delay", str(DELAY),
        "--retries", str(RETRIES),
        "--retry_delay", str(RETRY_DELAY),
        "--checkpoint_every", str(CHECKPOINT_EVERY),
    ]
    env = {**os.environ, "PYTHONPATH": "/workspace"}
    print(f"Running chunk → {out_filename} ({len(prompts_data)} prompts)", flush=True)
    subprocess.run(cmd, env=env, cwd="/workspace", check=True)

    if not os.path.exists(out_path):
        raise RuntimeError(f"Expected output missing: {out_path}")

    volume.commit()
    print(f"Done: {out_filename}", flush=True)
    return out_path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def chunk_list(lst: list, n: int) -> list[list]:
    k = (len(lst) + n - 1) // n
    return [lst[i * k:(i + 1) * k] for i in range(n) if lst[i * k:(i + 1) * k]]


@app.local_entrypoint()
def main():
    train_prompts = load_jsonl("data/prompts/split_train.jsonl")
    val_prompts   = load_jsonl("data/prompts/split_val.jsonl")
    test_prompts  = load_jsonl("data/prompts/split_test.jsonl")

    # build list of (prompts_data, out_filename) jobs
    jobs: list[tuple[list[dict], str]] = []

    # train: split into  parallel containers
    chunks = chunk_list(train_prompts, CONTAINERS)
    for i, chunk in enumerate(chunks):
        jobs.append((chunk, f"candidates_train_chunk{i}.jsonl"))

    # val and test can just use 1 container each
    jobs.append((val_prompts,  "candidates_val.jsonl"))
    jobs.append((test_prompts, "candidates_test.jsonl"))

    print(f"Launching {len(jobs)} Modal containers ({CONTAINERS} train chunks + val + test)...")
    print(f"Train prompts: {len(train_prompts)} → {CONTAINERS} chunks of ~{len(chunks[0])} each")

    # Run all jobs in parallel via map
    results = list(run_chunk.starmap(jobs))
    print("All containers finished:")
    for r in results:
        print(f"  {r}")

    print(f"\nMerge train chunks with:")
    chunk_files = " ".join(f"candidates_train_chunk{i}.jsonl" for i in range(CONTAINERS))
    print(f"  modal volume get {VOLUME_NAME} candidates_val.jsonl data/candidates/candidates_val.jsonl")
    print(f"  modal volume get {VOLUME_NAME} candidates_test.jsonl data/candidates/candidates_test.jsonl")
    for i in range(CONTAINERS):
        print(f"  modal volume get {VOLUME_NAME} candidates_train_chunk{i}.jsonl data/candidates/candidates_train_chunk{i}.jsonl")
    print(f"  cat data/candidates/candidates_train_chunk*.jsonl > data/candidates/candidates_train.jsonl")
