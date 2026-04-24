# OptiRAG

Automated RAG configuration optimization: **Stage 1** uses Optuna with a RAGAS scalar over FiQA (BEIR); **Stage 2** uses Deep Agents for prompt optimization. See the project plan document for design details.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Set GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_HOST
```

## CLI

```text
optirag data download
optirag index build --experiment experiments/fiqa_stage1.yaml
optirag eval run --experiment experiments/fiqa_stage1.yaml
optirag tune stage1 --experiment experiments/fiqa_stage1.yaml
optirag tune stage2 --experiment experiments/fiqa_stage1.yaml
```

## Develop

```bash
ruff check src tests
pytest tests/unit
```

Integration tests: `pytest tests/integration -m integration` (requires API keys).
