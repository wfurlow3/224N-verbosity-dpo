"""
Run generate_candidates.py on Modal.
"""
import os
import subprocess
import sys

import modal

APP_NAME = "verbosity-dpo-generate"
VOLUME_NAME = "verbosity-dpo-candidates"
SECRET_NAME = "moonshot-api-key"

# LOOK HERE CHANGE HERE FOR TRAIN/VAL/TEST
PROMPTS_FILE = "data/prompts/split_test.jsonl"
OUT_FILENAME = "candidates_test.jsonl"

WORKERS = 10
DELAY = 0.1
RETRIES = 2
RETRY_DELAY = 5.0
CHECKPOINT_EVERY = 1000 
TIMEOUT = 3600 * 5  # 5 hours

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
def run_generate():
    prompts_path = f"/workspace/{PROMPTS_FILE}"
    out_path = f"/out/{OUT_FILENAME}"

    cmd = [
        sys.executable, "-m", "src.data.generate_candidates",
        "--prompts", prompts_path,
        "--out", out_path,
        "--workers", str(WORKERS),
        "--delay", str(DELAY),
        "--retries", str(RETRIES),
        "--retry_delay", str(RETRY_DELAY),
        "--checkpoint_every", str(CHECKPOINT_EVERY),
    ]

    env = {**os.environ, "PYTHONPATH": "/workspace"}
    print(f"Running: {' '.join(cmd)}", flush=True)

    subprocess.run(cmd, env=env, cwd="/workspace", check=True)

    if not os.path.exists(out_path):
        raise RuntimeError(f"Expected output missing: {out_path}")

    volume.commit()
    print(f"Done. Output written to volume '{VOLUME_NAME}' as {OUT_FILENAME}.", flush=True)
    return out_path

@app.local_entrypoint()
def main():
    print("Send generate_candidates to Modal")
    out_path = run_generate.remote()
    print(f"Remote output: {out_path}")
    print("\nDownload by running")
    print(f"  modal volume get {VOLUME_NAME} {OUT_FILENAME} data/candidates/{OUT_FILENAME}")
