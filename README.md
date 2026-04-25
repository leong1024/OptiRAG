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
- **Quota-friendly subset** (`fiqa_max_docs`): if set in experiment YAML, only the first N FiQA corpus docs are loaded, and qrels/queries are filtered to match. Useful for free-tier quotas and faster iteration.

**Pinecone host resolution**

1. **Registry** `artifacts/pinecone_index_registry.json` (or `OPTIRAG_PINECONE_INDEX_REGISTRY_PATH`): map `"<dim>:<metric>"` → index host URL.
2. **Single host binding**: with an **empty** registry, set `PINECONE_INDEX_HOST`; the first `(dim, metric)` your runs need is recorded under that host. Further distinct pairs require new entries or `OPTIRAG_PINECONE_AUTO_CREATE=true`.
3. **Auto-create** (`OPTIRAG_PINECONE_AUTO_CREATE=true`): creates serverless indexes using `OPTIRAG_PINECONE_CLOUD` / `OPTIRAG_PINECONE_REGION` and stores hosts in the registry.

**Cost:** new fingerprints run full-corpus embedding and upsert; reuse the cache or cap `ragas.query_subset` for development.

### Step-by-step guide (FiQA + Optuna resume + plotting)

1. **Install and configure**
   - Create venv and install:
     - `python -m venv .venv`
     - `.venv\Scripts\activate`
     - `pip install -e ".[dev]"`
   - Copy env and set required keys:
     - `cp .env.example .env`
     - Set `GEMINI_API_KEY` and `PINECONE_API_KEY`.

2. **Prepare Pinecone host routing**
   - Choose one method:
     - Set `PINECONE_INDEX_HOST` (first `(dim,metric)` binding), or
     - Enable `OPTIRAG_PINECONE_AUTO_CREATE=true` and set `OPTIRAG_PINECONE_CLOUD` / `OPTIRAG_PINECONE_REGION`.
   - Registry is stored at `artifacts/pinecone_index_registry.json` by default.

3. **Prepare FiQA data**
   - Download data:
     - `optirag data download`

4. **Check experiment config**
   - Open `experiments/fiqa_stage1.yaml` and confirm:
     - `optuna.study_name` (stable name to resume),
     - `optuna.storage` (SQLite URL, e.g. `sqlite:///artifacts/optuna/optirag-fiqa-s1.db`),
     - `optuna.n_trials`,
     - `optuna.tune_index_hyperparams`:
       - `false` = retrieval-only tuning (cheaper),
       - `true` = full stage-1 tuning (chunk/embed/metric + retrieval knobs).

5. **Build index once (retrieval-only mode)**
   - If `tune_index_hyperparams: false`, run:
     - `optirag index build --experiment experiments/fiqa_stage1.yaml`

6. **Run baseline eval (recommended)**
   - `optirag eval run --experiment experiments/fiqa_stage1.yaml`

7. **Run Optuna tuning**
   - `optirag tune stage1 --experiment experiments/fiqa_stage1.yaml`
   - Trials are persisted in SQLite and per-trial JSON files under `artifacts/optuna/`.

8. **Resume after interruption (credits/manual stop)**
   - Re-run the same tune command with the same YAML.
   - As long as `optuna.storage` and `optuna.study_name` are unchanged, optimization continues from prior trials.

9. **Export CSV for plotting**
   - `optirag tune export-csv --experiment experiments/fiqa_stage1.yaml`
   - Default output: `artifacts/optuna/<study_name>_trials.csv`
   - Includes trial value, state, params, and user attrs (including saved RAGAS scores).

## CLI

```text
optirag data download
optirag index build --experiment experiments/fiqa_stage1.yaml
optirag index build --experiment experiments/fiqa_stage1.yaml --force
optirag eval run --experiment experiments/fiqa_stage1.yaml
optirag tune stage1 --experiment experiments/fiqa_stage1.yaml
optirag tune export-csv --experiment experiments/fiqa_stage1.yaml
optirag tune stage2 --experiment experiments/fiqa_stage1.yaml
```

## Develop

```bash
ruff check src tests
pytest tests/unit
```

Integration tests: `pytest tests/integration -m integration` (skipped unless `RUN_OPTIRAG_INTEGRATION=1`; needs API keys and network).
