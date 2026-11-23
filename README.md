# arbiter

decision models as warm starts for cross-encoder rerankers: a controlled study.

five public decision checkpoints built on ModernBERT-large (laya, beveridge, von, solvi, cortex), an nli checkpoint and a norm-matched noise perturbation are each used as the starting point for the same listwise cross-encoder fine-tuning on MS MARCO, at 250 to 16k training queries, and compared with plain ModernBERT-large. everything about the study lives in `configs/study.yaml`.

## setup

```
uv sync
```

checkpoints go in `models/<dir>` and datasets in `data/` (both gitignored); the directory names are listed in the config.

| dir | hugging face repo | revision |
|---|---|---|
| `ModernBERT-large` | answerdotai/ModernBERT-large | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` |
| `laya-en` | convaiinnovations/laya (subfolder laya-en) | `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1` |
| `Beveridge-Large-C` | sathiskumarjothi0/Beveridge-Large-C | `7cb0d24cce8572b8803a738c083b2419cdc40a8a` |
| `von` | mohdvasim/von | `8d072bdeb53523a1892d1382406e21c21c56fbda` |
| `solvi-large` | solvi-ai/solvi-large | `cd992673d838761c669611e2f59ba9661d7b1a15` |
| `cortex-1-large` | mukti-sys/cortex-1-large | `3fe2c0b1f997a82339dd13598759abee96495b6d` |
| `ModernBERT-large-nli` | tasksource/ModernBERT-large-nli | `ca476cb923a8637073d4ceb0f19f7fc236e260d4` |
| `minilm` | cross-encoder/ms-marco-MiniLM-L6-v2 | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| `bge-v2-m3` | BAAI/bge-reranker-v2-m3 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

data: `data/msmarco/msmarco-bm25/triplet-50/` (sentence-transformers/msmarco-bm25, shard 0), `data/msmarco/ms_marco/v1.1/` (validation split, used only to exclude overlapping queries), `data/trec-dl-2019|2020/` (official `qrels.txt` and `top1000.tsv.gz`), and `data/<beir set>/` for scifact, nfcorpus, fiqa and trec-covid (BEIR `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`).

## run order

```
uv run pytest                      # cpu tests: metrics vs ir_measures, leakage, accumulation, stats, io, plots
uv run arbiter prepare             # held-out splits, bm25 candidates, noise arm -> cache/
uv run arbiter smoke               # gpu: arms load, interrupt/resume, fp16 vs fp32, throughput
uv run arbiter run all             # grid -> select -> main -> extra, resumable
uv run arbiter status
uv run arbiter collect --confirmatory
uv run arbiter analyze
uv run arbiter plot
```

`arbiter run` refuses to start on uncommitted code or config changes, skips finished runs and resumes interrupted ones from their last checkpoint. test-set scores are written under `scores/sealed/` and only read by `collect --confirmatory` once every confirmatory run has finished.

## outputs

- `runs/<stage>/<run_id>/`: config, status, training log, raw dev and sealed test scores, kept weights
- `runs/selection.json`: learning rate chosen per arm on dev
- `results/`: per-query metrics, run table, `stats.json`, tables as csv and latex
- `figures/`: pdf and png
- `paper/`: the manuscript in two ieee formats (`journal/`, `conference/`), built with `tectonic paper/journal/main.tex`

run artifacts (configs, logs, statuses, per-query scores) are committed; model weights are not.

## preregistration

tag `prereg-v1` freezes the design, config and analysis code before any confirmatory run; `prereg-v1.1` records amendment 1 (wider learning-rate grid). later changes are listed in the paper.

## license

mit
