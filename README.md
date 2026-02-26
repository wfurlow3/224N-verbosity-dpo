# cs224n-verbosity-dpo

# DPO

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

File will be in `data/prompts/prompt_pool.jsonl`. By default, 50% of prompts are sampled from UltraFeedback and 50% from HelpSteer2. 5000 prompts were selected.

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
_For 5000 prompts -> 15000 candidates (13500 for train), we used Modal_

```bash
# val
python -m src.data.generate_candidates \
  --prompts data/prompts/split_val.jsonl \
  --out data/candidates/candidates_val.jsonl \
  --workers 4 \
  --delay 0.5
```

Generation takes a while, so switched to Modal.

### Candidate generation with Modal

`modal run --detach run_generate_modal.py` (from root; detach lets it run after terminal closes)

to download:

```bash
modal volume get verbosity-dpo-candidates candidates_train.jsonl data/candidates/candidates_train.jsonl
```

Note that 9 candidates weren't generated (prompts p000249, p001717, p001021) because the model deemed the prompts "high risk."

## 4. Validate candidates

- Uses LLM-as-a-judge (gpt-4o-mini) to filter for correct concise and verbose responses. Also filter out truncated responses or refusal responses.

```bash
python -m src.data.validate_candidates \
   --candidates data/candidates/candidates_train.jsonl \
   --prompts data/prompts/split_train.jsonl \
   --out data/candidates/candidates_train_train.jsonl
```

Train results:

- Validated: 9470 kept, 4021 dropped (truncated=2130, refusal=220, concise_judge=630, verbose_judge=1041)
- Concise kept: 2602
- Verbose kept: 2948
- Too short kept: 3920
- Wrote 9470 candidates to data/candidates/candidates_train_validated.jsonl

Val results:

- Validated: 532 kept, 215 dropped (truncated=122, refusal=15, concise_judge=25, verbose_judge=53)
- Concise kept: 151
- Verbose kept: 162
- Too short kept: 219
- Wrote 532 candidates to data/candidates/candidates_val_validated.jsonl

## 5. Build DPO data by pairing

```bash
python -m src.data.build_dpo \
  --candidates data/candidates/candidates_train_validated.jsonl \
  --prompts data/prompts/split_train.jsonl \
  --out data/dpo/dpo_train.jsonl
```

Train: 4561 DPO pairs
Val: 256 DPO pairs

## Next steps
