# SFT on Modal

## Compute and VRAM
- `A100-40GB`

## Runtime requirements
- CUDA-enabled environment with `bitsandbytes` support.
- `accelerate` configured in the container.
- HF auth already handled via `huggingface-cli login` (no `HF_TOKEN` needed in `.env`).

## Accelerate config checklist
- Distributed type: `NO` (single GPU to start).
- Mixed precision: `fp16` or `bf16` 
- Gradient accumulation


