# Logistics QA — LoRA fine-tuning of Qwen 2.5-7B-Instruct

LoRA fine-tuning of [Qwen 2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on a 12K-example synthetic Q&A dataset covering logistics and supply-chain operations: freight calculations, carrier performance, claims, routing, SOPs, and FMCSA / customs compliance. Includes the dataset generator, training pipeline, evaluation harness, and a FastAPI streaming inference server.

```text
Question:  "What's the deadline for filing a concealed-damage freight claim under U.S. law?"
Answer:    "Under 49 CFR 370.3, the claim must be filed in writing with the carrier
            within 9 months of delivery. The carrier must acknowledge within 30 days
            and pay, decline, or make a firm offer within 120 days."
```

## Why this project

While doing logistics-ops analytics during a co-op term, I noticed a recurring pattern: the same regulatory and procedural questions came up over and over — "what's a Carmack-Amendment claim limit?", "what's the DIM factor for LTL?", "what's the difference between C-TPAT and AEO?". Generic LLMs answer these inconsistently because the underlying facts are scattered across CFR text, NMFC tariffs, and trade press. This project tests whether a small LoRA fine-tune on a curated synthetic corpus can move a 7B model meaningfully closer to expert reliability on domain Q&A.

## Architecture

```
                                 ┌────────────────────────┐
   data/seed_topics.json  ────►  │  prepare_dataset.py    │ ── Anthropic API ──► 12K Q&A pairs
                                 └────────────────────────┘
                                              │
                                              ▼
                                 train.jsonl / val.jsonl / test.jsonl
                                              │
                                              ▼
                                 ┌────────────────────────┐       Weights &
   Qwen2.5-7B-Instruct  ──────►  │  src/train.py          │ ────► Biases
   (4-bit, NF4)                  │  LoRA r=16, α=32        │
                                 │  PEFT + TRL              │
                                 └────────────────────────┘
                                              │
                                              ▼
                                  artifacts/checkpoints/final  (LoRA adapter, ~80 MB)
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                ▼                             ▼                             ▼
       src/evaluate.py              server/app.py (FastAPI)          notebooks/03_evaluate.ipynb
       EM + token-F1 +              /generate                          base vs fine-tuned
       per-category                 /generate/stream (SSE)             comparison plots
```

## Results

> **Note:** Replace the placeholder numbers below after your first training run. Run `./scripts/eval.sh --compare --output artifacts/results/eval.json` and paste the printed summary into this section.

| Model                       | Exact Match | Token F1 |
| --------------------------- | ----------- | -------- |
| Qwen 2.5-7B-Instruct (base) | _TBD_       | _TBD_    |
| + LoRA (this work)          | _TBD_       | _TBD_    |
| **Δ**                       | _TBD_       | _TBD_    |

Per-category breakdown is generated in the same eval JSON.

## Setup

### Option A — Google Colab (easiest, free T4)

The three notebooks in `notebooks/` run the full workflow on Colab's free T4. Open each in order:

1. `01_generate_dataset.ipynb` — generate the 12K dataset (~$5-15 in Anthropic credits, ~30 min)
2. `02_train_lora.ipynb` — train the adapter (~3-6 hours on T4)
3. `03_evaluate.ipynb` — produce side-by-side metrics and plots

### Option B — Local

```bash
git clone https://github.com/<you>/logistics-qa-lora.git
cd logistics-qa-lora

# Python 3.10+ required
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For GPU users, install the CUDA-matching torch wheel:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# Copy env template and fill in your keys
cp .env.example .env
# edit .env, add ANTHROPIC_API_KEY at minimum
```

## Workflow

### 1. Generate the dataset

```bash
# Smoke test — generates 1 batch per category (~50 examples), verifies the pipeline
python -m data.prepare_dataset --smoke

# Full run — generates 12K examples and writes train/val/test splits
python -m data.prepare_dataset --target 12000 --batch-size 8 --split
```

The script is resumable. If interrupted, rerunning skips already-generated examples (matched by question hash) and continues until the target is reached.

### 2. Train

```bash
# Default 3-epoch QLoRA run, ~3-6 hours on a T4
./scripts/train.sh

# Or pass overrides
./scripts/train.sh --epochs 5 --lr 3e-4 --batch-size 8
```

The adapter is saved to `artifacts/checkpoints/final/`. With QLoRA, the adapter alone is ~80 MB — small enough to upload to the Hub.

### 3. Evaluate

```bash
./scripts/eval.sh
```

Prints a summary table and writes full per-example predictions to `artifacts/results/eval.json` for inspection.

### 4. Serve

```bash
# With the trained adapter
ADAPTER_PATH=artifacts/checkpoints/final uvicorn server.app:app --port 8000

# Without a model (mock backend — useful for testing the HTTP layer)
USE_MOCK=true uvicorn server.app:app --port 8000

# Or via Docker
docker build -t logistics-qa-lora .
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e ADAPTER_PATH=/app/artifacts/checkpoints/final \
  logistics-qa-lora
```

Endpoints:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the DIM factor for LTL shipments?"}'
curl -N -X POST localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain Carmack liability."}'
```

## Configuration

Everything tunable lives in [`src/config.py`](src/config.py). The most-changed knobs:

| Setting               | Default                  | Where             |
| --------------------- | ------------------------ | ----------------- |
| Base model            | `Qwen/Qwen2.5-7B-Instruct` | `MODEL.name`      |
| LoRA rank             | 16                       | `LORA.r`          |
| LoRA target modules   | q,k,v,o proj             | `LORA.target_modules` |
| Effective batch size  | 16 (4 × 4 accum)         | `TRAIN.*`         |
| Learning rate         | 2e-4                     | `TRAIN.learning_rate` |
| Max sequence length   | 1024                     | `MODEL.max_seq_length` |

To target a different model (Llama-3-8B, Mistral-7B, etc.), change `MODEL.name`. The rest of the pipeline is model-agnostic — any HF causal-LM model with a chat template will work.

## Tests

```bash
USE_MOCK=true pytest -v
```

Tests cover dataset IO, chat-template label masking, evaluation metrics, and the full FastAPI server (using the mock backend, so no GPU required).

## Security

This project requires API keys. **Read [`SECURITY.md`](SECURITY.md) before pushing the repo public.** Short version:

1. Set a monthly spending cap on the Anthropic console *before* generating a key.
2. Confirm `.env` is git-ignored (`git check-ignore .env` should print `.env`).
3. Install the pre-commit hook: `pip install pre-commit && pre-commit install`. This runs `gitleaks` on every commit.

The `.gitignore` excludes `.env`, all credential file patterns, checkpoint files, and the generated dataset.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Alibaba Cloud) — base model
- [PEFT](https://github.com/huggingface/peft) and [TRL](https://github.com/huggingface/trl) — LoRA + SFT infrastructure
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [Anthropic Claude](https://www.anthropic.com/claude) — synthetic dataset generation
