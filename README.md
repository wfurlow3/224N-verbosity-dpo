# cs224n-verbosity-dpo

Verbosity control via data-centric DPO. We construct a verbosity-disentangled preference dataset and train Mistral-7B-Instruct-v0.2 with DPO to produce concise, correct responses.

## Project Structure

```
├── configs/                             # Evaluation configs for AlpacaEval
├── data/
│   ├── alpacaeval/                      # AlpacaEval outputs and annotations
│   ├── candidates/                      # Raw candidates + filtered candidates
│   │   └── raw/                         # Raw chunked candidates from Modal
│   ├── dpo/
│   │   ├── disentangled/                # Verbosity-disentangled DPO pairs
│   │   └── vanilla/                     # UltraFeedback binarized DPO pairs
│   ├── prompts/                         # Prompt pool and train/val/test splits
├── notebooks/                           # Plots
├── scripts/
│   ├── data/                            # Data pipeline scripts
│   ├── dev/                             # Development/smoke test scripts
│   ├── eval/                            # AlpacaEval generation + scoring
│   └── train/                           # SFT + DPO training scripts (Modal)
│       ├── dpo_train_qlora.py           # DPO (disentangled or vanilla, with or without SFT)
│       ├── sft_train_qlora.py           # SFT
│       └── simpo_train_qlora.py         # SimPO
├── sft_model/                           # Local SFT adapter files
└── src/
    ├── data/                            # Scripts for building disentangled and vanilla datasets for DPO.
    └── teacher/                         # Sets up Kimi API client (for teacher model to generate responses)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
huggingface-cli login
```

---

## Data Pipeline

### 1. Build prompt pool

Sample 10,000 prompts equally from UltraFeedback and HelpSteer2:

```bash
python -m src.data.build_prompt_pool \
  --datasets ultrafeedback helpsteer2 \
  --n_prompts 10000 \
  --frac_ultrafeedback 0.5 \
  --seed 42 \
  --out data/prompts/prompt_pool.jsonl
```

### 2. Split prompts (90/5/5 train/val/test)

```bash
python -m src.data.split_prompts \
  --prompts data/prompts/prompt_pool.jsonl \
  --out_dir data/prompts \
  --train_frac 0.90 --val_frac 0.05
```

Creates `split_train.jsonl`, `split_val.jsonl`, `split_test.jsonl` in `data/prompts/`.
Split: 9,000 train / 500 val / 500 test prompts.

### 3. Generate candidates (Modal)

Generates 3 style variants per prompt (concise, verbose, too_short) using Kimi (moonshot-v1-8k).
note: Kimi refuses some prompts it deems "high-risk", so not all candidates are produced.

```bash
modal run --detach scripts/data/run_generate_modal.py
```

Download from Modal volume after completion:

```bash
modal volume get verbosity-dpo-candidates candidates_train.jsonl data/candidates/candidates_train.jsonl
modal volume get verbosity-dpo-candidates candidates_val.jsonl data/candidates/candidates_val.jsonl
modal volume get verbosity-dpo-candidates candidates_test.jsonl data/candidates/candidates_test.jsonl
```

**Results:** 26,979 train / 1,497 val / 1,500 test candidates generated.

### 4. Validate candidates

Filters truncated and refusal responses heuristically, then uses GPT-4o-mini as a judge toensure correctness for concise and verbose variants. Too-short responses are not judged for correctness.

```bash
# Train
python -m src.data.validate_candidates \
  --candidates data/candidates/candidates_train.jsonl \
  --prompts data/prompts/split_train.jsonl \
  --out data/candidates/candidates_train_validated.jsonl

# Val
python -m src.data.validate_candidates \
  --candidates data/candidates/candidates_val.jsonl \
  --prompts data/prompts/split_val.jsonl \
  --out data/candidates/candidates_val_validated.jsonl

# Test
python -m src.data.validate_candidates \
  --candidates data/candidates/candidates_test.jsonl \
  --prompts data/prompts/split_test.jsonl \
  --out data/candidates/candidates_test_validated.jsonl
```

**Train:** 18,886 kept / 8,093 dropped (truncated=229, refusal=506, concise_judge=3,531, verbose_judge=3,827)

- Concise: 5,261 | Verbose: 4,921 | Too-short: 8,704

**Val:** 1,178 kept / 319 dropped (truncated=15, refusal=29, concise_judge=123, verbose_judge=152)

- Concise: 361 | Verbose: 331 | Too-short: 486

**Test:** 1,189 kept / 311 dropped (truncated=17, refusal=34, concise_judge=121, verbose_judge=139)

- Concise: 363 | Verbose: 344 | Too-short: 482

### 5. Build disentangled DPO pairs

Pairs concise (chosen) vs verbose and too-short (rejected):

```bash
python -m src.data.build_dpo \
  --candidates data/candidates/candidates_train_validated.jsonl \
  --prompts data/prompts/split_train.jsonl \
  --out data/dpo/disentangled/dpo_train.jsonl

python -m src.data.build_dpo \
  --candidates data/candidates/candidates_val_validated.jsonl \
  --prompts data/prompts/split_val.jsonl \
  --out data/dpo/disentangled/dpo_val.jsonl

python -m src.data.build_dpo \
  --candidates data/candidates/candidates_test_validated.jsonl \
  --prompts data/prompts/split_test.jsonl \
  --out data/dpo/disentangled/dpo_test.jsonl
```

**Results:** 8,734 train / 640 val / 643 test pairs

### 6. Build vanilla DPO pairs

Sample from UltraFeedback Binarized (no verbosity control):

```bash
python -m src.data.build_vanilla_dpo \
  --n_train 8000 \
  --n_val 640 \
  --seed 42 \
  --out_dir data/dpo/vanilla
```

**Results:** 8,000 train / 640 val pairs

### 7. Prepare SFT data

Converts the disentangled DPO pairs into SFT format, using the concise (chosen) responses as supervised outputs. Deduplicates by prompt.

```bash
python scripts/data/convert_dpo_to_sft.py
```

Reads `data/dpo/disentangled/dpo_train.jsonl` + `dpo_val.jsonl` → `data/sft/train.json`
Reads `data/dpo/disentangled/dpo_test.jsonl` → `data/sft/val.json`

### 8. Upload data to Modal volume

```bash
modal volume put verbosity-data data/dpo/disentangled/dpo_train.jsonl /dpo/disentangled/dpo_train.jsonl
modal volume put verbosity-data data/dpo/disentangled/dpo_val.jsonl /dpo/disentangled/dpo_val.jsonl
modal volume put verbosity-data data/dpo/vanilla/dpo_train.jsonl /dpo/vanilla/dpo_train.jsonl
modal volume put verbosity-data data/dpo/vanilla/dpo_val.jsonl /dpo/vanilla/dpo_val.jsonl
modal volume put verbosity-data data/sft/train.json /sft/train.json
modal volume put verbosity-data data/sft/val.json /sft/val.json
```

---

## Training

All training runs on Modal (H100). Base model: Mistral-7B-Instruct-v0.2.
QLoRA: 4-bit NF4, LoRA rank 16, α=32, dropout 0.05. All runs: lr=1e-5, 1000 steps.

### SFT

```bash
modal run --detach scripts/train/sft_train_qlora.py
```

Download adapter:

```bash
modal volume get verbosity-outputs sft_model /tmp/sft_model
```

### DPO (β=0.1)

All four DPO conditions use a single script. The `--data` flag selects the dataset and `--sft` loads the SFT adapter before training:

```bash
# Disentangled DPO — instruct base
modal run --detach scripts/train/dpo_train_qlora.py --data disentangled

# Disentangled DPO — SFT
modal run --detach scripts/train/dpo_train_qlora.py --data disentangled --sft

# Vanilla DPO — instruct base
modal run --detach scripts/train/dpo_train_qlora.py --data vanilla

# Vanilla DPO — SFT
modal run --detach scripts/train/dpo_train_qlora.py --data vanilla --sft
```

### SimPO — instruct base (β=2.5, γ=0.25, lr=5e-7)

```bash
modal run --detach scripts/train/simpo_train_qlora.py
```

---

## Evaluation

### 1. Generate AlpacaEval outputs (Modal)

```bash
modal run scripts/eval/modal_run_gen_alpacaeval.py \
  --adapter_dir /root/repo/outputs/<model_dir> \
  --output_path data/alpacaeval/<model_name>_outputs.json \
  --generator_name <model_name>
```

download outputs:

```bash
modal volume get verbosity-outputs <model_name>_outputs.json data/alpacaeval/<model_name>_outputs.json
```

### 2. Score with AlpacaEval 2.0

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your_openai_api_key"
alpaca_eval evaluate \
  --model_outputs data/alpacaeval/<model_name>_outputs.json \
  --annotators_config $(pwd)/configs/alpaca_eval_gpt4omini.yaml \
  --output_path data/alpacaeval/<model_name>/
```

Note: need to edit `--output_path` per model to avoid overwriting shared `annotations.json`.

---

## Results

| Model                          | WR (%) | LC-WR (%) | Avg. Length |
| ------------------------------ | ------ | --------- | ----------- |
| Mistral-7B-Instruct (baseline) | 20.0   | 28.1      | 1381        |
| + SFT                          | 5.2    | 3.6       | 2004        |
| + Vanilla DPO                  | 11.80  | 14.54     | 1488        |
| + SFT + Vanilla DPO            | 11.68  | 14.55     | 1480        |
| + Disentangled DPO             | --     | --        | --          |
| + SFT + Disentangled DPO       | --     | --        | --          |
| + SimPO                        | --     | --        | --          |
