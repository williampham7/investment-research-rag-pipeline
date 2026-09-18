# Retrieval evaluation results

67 hand-labeled queries, paper-level relevance, retrieval depth = 10.

## All runs

| Config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| section-aware + bge-large + hybrid + rerank (baseline) | 1.000 | 1.000 | 1.000 |
| fixed-size + bge-large + hybrid + rerank | 1.000 | 1.000 | 1.000 |
| section-aware + e5-large + hybrid + rerank | 1.000 | 1.000 | 1.000 |
| section-aware + bge-large + dense-only + rerank | 1.000 | 1.000 | 1.000 |
| section-aware + bge-large + hybrid + no-rerank | 1.000 | 1.000 | 1.000 |

## Ablations (baseline vs. one changed variable)

### Chunking strategy

| Config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| section-aware + bge-large + hybrid + rerank (baseline) | 1.000 | 1.000 | 1.000 |
| fixed-size + bge-large + hybrid + rerank | 1.000 | 1.000 | 1.000 |

### Embedding model

| Config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| section-aware + bge-large + hybrid + rerank (baseline) | 1.000 | 1.000 | 1.000 |
| section-aware + e5-large + hybrid + rerank | 1.000 | 1.000 | 1.000 |

### Retrieval mode

| Config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| section-aware + bge-large + hybrid + rerank (baseline) | 1.000 | 1.000 | 1.000 |
| section-aware + bge-large + dense-only + rerank | 1.000 | 1.000 | 1.000 |

### Reranking

| Config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| section-aware + bge-large + hybrid + rerank (baseline) | 1.000 | 1.000 | 1.000 |
| section-aware + bge-large + hybrid + no-rerank | 1.000 | 1.000 | 1.000 |
