# cs224n-verbosity-dpo

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

## SFT skeleton (no auto-training)

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

## What you must add for this to work

- `OPENAI_API_KEY` with API billing enabled is required for AlpacaEval judging.
- ChatGPT Plus does not include OpenAI API billing/access.
- AlpacaEval uses an LLM judge and will incur API cost.
- Modal GPU setup will be needed later for actual SFT/DPO training runs.
