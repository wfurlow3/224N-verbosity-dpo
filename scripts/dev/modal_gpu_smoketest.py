import modal

app = modal.App("gpu-smoketest")
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="T4", image=image, timeout=600)
def check_gpu():
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

@app.local_entrypoint()
def main():
    print(check_gpu.remote())
