# arbiter

benchmark of laya decision models against cross-encoder rerankers for zero-shot passage reranking.

## requirements

- python 3.11 to 3.13
- [uv](https://docs.astral.sh/uv/)
- cuda gpu or cpu (picked automatically)
- about 12 gb disk for models and data

## run

```
make setup
make fetch
make all
```

`make all` is resumable. scores and analyses are cached in `.cache/` and skipped on rerun.

## steps

| target | does |
|---|---|
| `setup` | install dependencies |
| `fetch` | download models into `models/` and datasets into `data/` |
| `smoke` | score one pair with every model |
| `score` | rerank all datasets with all models |
| `benchmark` | quality, latency, memory, bootstrap intervals |
| `cascade` | decision model filter before a cross-encoder |
| `calibration` | ece, brier, reliability |
| `stability` | rank stability under typos, synonyms, paraphrases |
| `longcontext` | long passage test on locomo |
| `prompts` | rubric and instruction wording |
| `layout` | query in the state or in the question |
| `validity` | random order floor, full precision rerun, recalibration |
| `figures` `tables` | render `plots/` and `tables/` |
| `verify` | run the tests |

## device

```
uv run python -m decision_rerank.benchmark --device cpu
```

## layout

```
CONFIG.yaml        models, datasets, sizes, seeds
src/decision_rerank/
tests/
results/           json and csv outputs
plots/  tables/
```

## data

scifact and nfcorpus (beir), trec dl 2019 and 2020, locomo, wordnet. all fetched by `make fetch`.

## models

laya-en, laya-multi, laya-td, bge-reranker-v2-m3, bge-reranker-base, qwen3-reranker-0.6b, jina-reranker-v2, ms-marco-minilm-l6, plus bge-small-en-v1.5 and qwen2.5-1.5b-instruct as support. repositories are listed in `CONFIG.yaml`.
