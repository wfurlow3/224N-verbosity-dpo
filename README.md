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
We mainly used this script for testing.

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

Note that some candidates because the model deemed the prompts "high risk."
26979 train candidates generated
1497 val candidates generated
1500 test candidates generated

## 4. Validate candidates

- Uses LLM-as-a-judge (gpt-4o-mini) to filter for correct concise and verbose responses. Also filter out truncated responses or refusal responses.

```bash
python -m src.data.validate_candidates \
   --candidates data/candidates/candidates_train.jsonl \
   --prompts data/prompts/split_train.jsonl \
   --out data/candidates/candidates_train_train.jsonl
```

Train results:

Validated: 18886 kept, 8093 dropped (truncated=229, refusal=506, concise_judge=3531, verbose_judge=3827)
Concise kept: 5261
Verbose kept: 4921
Too short kept: 8704

Wrote 18886 candidates to data/candidates/candidates_train_validated.jsonl

Val results:
Validated: 1178 kept, 319 dropped (truncated=15, refusal=29, concise_judge=123, verbose_judge=152)
Concise kept: 361
Verbose kept: 331
Too short kept: 486

Wrote 1178 candidates to data/candidates/candidates_val_validated.jsonl

Train results:
Validated: 1189 kept, 311 dropped (truncated=17, refusal=34, concise_judge=121, verbose_judge=139)
Concise kept: 363
Verbose kept: 344
Too short kept: 482
Wrote 1189 candidates to data/candidates/candidates_test_validated.jsonl

## 5. Build DPO data by pairing

```bash
python -m src.data.build_dpo \
  --candidates data/candidates/candidates_train_validated.jsonl \
  --prompts data/prompts/split_train.jsonl \
  --out data/dpo/dpo_train.jsonl
```

Train: 8734 DPO pairs
Val: 640 DPO pairs
Test: 643 DPO pairs

## Next steps

Scaffold for data-centric verbosity control experiments (SFT -> DPO), with AlpacaEval generation/evaluation wiring and QLoRA SFT skeleton scripts.

## Setup

```bash
cd /Users/wfurlow/Desktop/224N-verbosity-dpo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

HF auth is expected to be pre-configured with:

```bash
huggingface-cli login
```

## Generate AlpacaEval outputs (Mistral-7B-Instruct, deterministic)

```bash
source .venv/bin/activate
python scripts/eval/gen_alpacaeval_outputs.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --output_path data/alpacaeval/mistral_7b_instruct_v02_outputs.json
```

Output format is a JSON list of:
- `instruction`
- `output`
- `generator`

## Run AlpacaEval

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your_openai_api_key"
bash scripts/eval/run_alpacaeval.sh \
  data/alpacaeval/mistral_7b_instruct_v02_outputs.json \
  mistral-7b-instruct-v0.2-temp0 \
  configs/alpaca_eval_gpt4omini.yaml
```

Note: we override AlpacaEval's default `chatgpt_fn` judge config in-repo to avoid deprecated OpenAI judge model versions.

## SFT skeleton 

Prepare SFT pairs (`instruction`, `output`) from OASST1:

```bash
source .venv/bin/activate
python scripts/sft/prepare_sft_data.py \
  --dataset_name OpenAssistant/oasst1 \
  --split train \
  --output_path data/sft/train.jsonl
```

Dry-run QLoRA scaffold (does not train unless `--run_train` is passed):

```bash
source .venv/bin/activate
python scripts/sft/train_sft_qlora.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --data_path data/sft/train.jsonl \
  --output_dir outputs/sft-qlora-mistral7b \
  --max_seq_len 2048 \
  --steps 1000 \
  --batch_size 1 \
  --lr 2e-4
```

# Next steps

goals:

- raw instruct
- instruct + sft
- instruct + vanilla dpo
- instruct + disentangled dpo
- instruct + sft + vanilla dpo
- instruct + sft + disentangled dpo
