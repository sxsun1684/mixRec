"""
evaluate.py

Model-agnostic retrieval evaluation framework.

Supports:

- Recall@K
- NDCG@K
- MRR

Any retriever implementing:

    search(query, k) -> [{"id": ...}, ...]

can be evaluated with this module.

This allows direct comparison of lexical, semantic,
hybrid, and reranked retrieval systems using the same
benchmark dataset.

qrels format:

{
    query: {relevant_doc_id_1, relevant_doc_id_2, ...}
}
"""

import math


def _dcg(rels):
    """
    Compute Discounted Cumulative Gain (DCG).

    Relevant results appearing higher in the ranking
    contribute more heavily than those appearing later.
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def evaluate(retriever, qrels, k=10):
    """
    Evaluate a retriever on a benchmark dataset.

    Returns average Recall@K, NDCG@K, and MRR
    across all evaluation queries.
    """
    recalls, ndcgs, rrs = [], [], []
    for query, relevant in qrels.items():
        retrieved = [h["id"] for h in retriever.search(query, k=k)]

        # Binary relevance labels for retrieved results.
        rels = [1 if rid in relevant else 0 for rid in retrieved]

        n_rel = len(relevant)

        # Recall@K:
        # Fraction of relevant documents retrieved
        # within the top-K results.
        recalls.append(sum(rels) / n_rel if n_rel else 0.0)

        # NDCG@K:
        # Ranking quality normalized by the ideal ranking.
        idcg = _dcg([1] * min(n_rel, k))
        ndcgs.append(_dcg(rels) / idcg if idcg else 0.0)

        # Mean Reciprocal Rank (MRR):
        # Reciprocal rank of the first relevant result.
        rr = 0.0
        for i, r in enumerate(rels):
            if r:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)

    n = len(qrels)

    # Aggregate metrics across all evaluation queries.
    return {
        f"Recall@{k}": sum(recalls) / n,
        f"NDCG@{k}": sum(ndcgs) / n,
        "MRR": sum(rrs) / n,
    }

def compare(retrievers, qrels, k=10):
    """
    Evaluate multiple retrievers on the same benchmark
    and print a side-by-side comparison table.

    Parameters
    ----------
    retrievers : dict
        Mapping from retriever name to retriever instance.
    """
    rows = {name: evaluate(r, qrels, k) for name, r in retrievers.items()}
    cols = list(next(iter(rows.values())).keys())

    print(f"{'retriever':<16}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (16 + 12 * len(cols)))

    for name, m in rows.items():
        print(f"{name:<16}" + "".join(f"{m[c]:>12.3f}" for c in cols))

    return rows