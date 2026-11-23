# arbiter

decision models as warm starts for cross-encoder rerankers: a controlled study.

five public decision checkpoints built on ModernBERT-large (laya, beveridge, von, solvi, cortex), an nli checkpoint and a norm-matched noise perturbation are each used as the starting point for the same listwise cross-encoder fine-tuning on MS MARCO, at 250 to 16k training queries, and compared with plain ModernBERT-large. everything about the study lives in `configs/study.yaml`.

## setup

```
uv sync
```

checkpoints go in `models/<dir>` and datasets in `data/` (both gitignored); the directory names are listed in the config.

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
