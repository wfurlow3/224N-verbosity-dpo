# cs224n-verbosity-dpo

## 1. Build prompt pool with (samples, does not iterate full datasets):

```bash
# run from the root (half of data from ultrafeedback, half from helpsteer2)
python -m src.data.build_prompt_pool \
  --datasets ultrafeedback helpsteer2 \
  --n_prompts 5000 \
  --frac_ultrafeedback 0.5 \
  --seed 42 \
  --out data/prompts/prompt_pool.jsonl
```

File will be in `data/prompts/prompt_pool.jsonl`. By default, 50% of prompts are sampled from UltraFeedback and 50% from HelpSteer2 (adjust with `--frac_ultrafeedback`).

## 2. Split prompts so train/val/test (90/5/5) don’t share the same prompts:

```bash
python -m src.data.split_prompts \
  --prompts data/prompts/prompt_pool.jsonl \
  --out_dir data/prompts \
  --train_frac 0.90 --val_frac 0.05
```

Creates `split_train.jsonl`, `split_val.jsonl`, `split_test.jsonl` in data/prompts

## 3. Generate 3 candidates for each prompt

The candidates are **concise**, **verbose**, and **too_short**. Uses the teacher (Kimi) API.

Other options: `--workers N` for concurrent calls. `--retries` and `--retry_delay` control retries on failure.

```bash
# train
python -m src.data.generate_candidates \
  --prompts data/prompts/split_train.jsonl \
  --out data/candidates/candidates_train.jsonl \
  --workers 4 \
  --delay 0.1
```

```bash
# val
python -m src.data.generate_candidates \
  --prompts data/prompts/split_val.jsonl \
  --out data/candidates/candidates_val.jsonl \
  --workers 4 \
  --delay 0.5
```

```bash
# test
python -m src.data.generate_candidates \
  --prompts data/prompts/split_test.jsonl \
  --out data/candidates/candidates_test.jsonl \
  --workers 4 \
  --delay 0.5
```

## Next steps

- **Validate candidates:** `python -m src.data.validate_candidates` (clean out bad candidates).
- **Label pairs:** Use LLM judge to verify (a) concise is correct, (b) verbose is correct, (c) too-short misses key info and is incorrect, and drop any failures.
- **Build DPO data:** `python -m src.data.build_dpo` → `dpo_train.jsonl` / `dpo_val.jsonl`.
- **Train DPO**
