# OptiRAG

Automated RAG configuration optimization: **Stage 1** uses Optuna with a RAGAS scalar over FiQA (BEIR); **Stage 2** uses Deep Agents for prompt optimization. See the project plan document for design details.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Set GEMINI_API_KEY, PINECONE_API_KEY, and Pinecone index access (see Stage 1 below)
```

## Stage 1 end-to-end

Each **index-defining** configuration (chunking, cleaning, embedding model, output dim, L2 norm, Pinecone metric) gets a **cache fingerprint** under `artifacts/index_cache/`. Vectors are stored in **one physical Pinecone index per (vector dimension, metric)**, with a **namespace per fingerprint** so different chunk/embed settings do not collide.

- **Retrieval-only tuning** (`optuna.tune_index_hyperparams: false` in the experiment YAML): Optuna varies `top_k`, score cut, dedup, context budget, etc. The **frozen** index settings come from `stage1_base` optional keys (merged onto defaults) in the experiment file. Run `optirag index build` once with the same YAML so the cache matches.
- **Full Stage 1 tuning** (`tune_index_hyperparams: true`): each trial may change chunk/embed/metric; `optirag tune stage1` **embeds and upserts** per new fingerprint (cached on repeat) and uses a matching **query embedder** for that trial.

**Pinecone host resolution**

1. **Registry** `artifacts/pinecone_index_registry.json` (or `OPTIRAG_PINECONE_INDEX_REGISTRY_PATH`): map `"<dim>:<metric>"` → index host URL.
2. **Single host binding**: with an **empty** registry, set `PINECONE_INDEX_HOST`; the first `(dim, metric)` your runs need is recorded under that host. Further distinct pairs require new entries or `OPTIRAG_PINECONE_AUTO_CREATE=true`.
3. **Auto-create** (`OPTIRAG_PINECONE_AUTO_CREATE=true`): creates serverless indexes using `OPTIRAG_PINECONE_CLOUD` / `OPTIRAG_PINECONE_REGION` and stores hosts in the registry.

**Cost:** new fingerprints run full-corpus embedding and upsert; reuse the cache or cap `ragas.query_subset` for development.

## CLI

```text
optirag data download
optirag index build --experiment experiments/fiqa_stage1.yaml
optirag index build --experiment experiments/fiqa_stage1.yaml --force
optirag eval run --experiment experiments/fiqa_stage1.yaml
optirag tune stage1 --experiment experiments/fiqa_stage1.yaml
optirag tune stage2 --experiment experiments/fiqa_stage1.yaml
```

## Develop

```bash
ruff check src tests
pytest tests/unit
```

Integration tests: `pytest tests/integration -m integration` (skipped unless `RUN_OPTIRAG_INTEGRATION=1`; needs API keys and network).
