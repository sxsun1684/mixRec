"""
retrieval.py

Schema-driven retrieval framework.

Design goals:

    - Data independent
    - Model independent
    - Index independent

Architecture:

    Schema
        ↓
    Corpus
        ↓
    Embedder
        ↓
    VectorIndex
        ↓
    Retriever

Switching from TF-IDF to SentenceTransformer requires
no changes to retrieval logic.
"""
from __future__ import annotations

import re
import json
import hashlib
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np

# Registered field cleaners used during text construction.
# New cleaning rules can be added without modifying schema logic.
# ============================================================
# 1) Schema
#
# Declarative configuration describing how raw records are
# transformed into searchable documents and metadata.
# ============================================================

# Registered field cleaners used during text construction.
# New cleaning rules can be added without modifying schema logic.
CLEANERS: dict[str, Callable[[Any], str]] = {
    "identity": lambda s: "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s),
    "html": lambda s: re.sub(
        r"\s+",
        " ",
        re.sub(r"<[^>]+>", " ", str(s)).replace("**", " ")
    ).strip()
    if isinstance(s, str) else "",
}


@dataclass
class FieldSpec:
    """
    Configuration for a searchable field.

    Defines:
        - source column
        - display label
        - cleaning strategy
    """
    field: str
    label: str = ""
    clean: str = "identity"


@dataclass
class Schema:
    """
    Declarative schema describing how records are converted
    into retrieval documents and metadata.
    """

    id_field: str

    # Fields included in searchable text.
    text_spec: list[FieldSpec]

    # Fields retained for filtering and result display.
    metadata_fields: list[str] = field(default_factory=list)

    def build_text(self, record: dict) -> str:
        """
        Build retrieval text from multiple fields according
        to schema configuration.
        """
        parts = []

        for fs in self.text_spec:
            cleaner = CLEANERS.get(
                fs.clean,
                CLEANERS["identity"]
            )

            value = cleaner(
                record.get(fs.field)
            )

            if value:
                parts.append(
                    f"{fs.label}: {value}"
                    if fs.label
                    else value
                )

        return "\n".join(parts)

    def version(self, model_name: str) -> str:
        """
        Generate a schema-model fingerprint used for cache
        invalidation and index versioning.
        """
        payload = json.dumps(
            {
                "text_spec": [
                    asdict(f)
                    for f in self.text_spec
                ],
                "model": model_name,
            },
            sort_keys=True,
        )

        return hashlib.md5(
            payload.encode()
        ).hexdigest()[:8]

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        """
        Create a schema from a JSON-compatible configuration.
        """
        return cls(
            id_field=d["id_field"],
            text_spec=[
                FieldSpec(**fs)
                for fs in d["text_spec"]
            ],
            metadata_fields=d.get(
                "metadata_fields",
                []
            ),
        )

# ============================================================
# 2) Corpus
#
# Convert heterogeneous data sources (CSV, JSONL, records)
# into a unified document representation for retrieval.
# ============================================================

@dataclass
class Corpus:
    ids: list
    texts: list[str]
    metadata: list[dict]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def from_records(cls, records: list[dict], schema: Schema) -> "Corpus":
        ids, texts, meta = [], [], []
        for r in records:
            ids.append(r.get(schema.id_field))
            texts.append(schema.build_text(r))
            meta.append({k: r.get(k) for k in schema.metadata_fields})
        return cls(ids, texts, meta)

    @classmethod
    def from_csv(cls, path: str, schema: Schema) -> "Corpus":
        import pandas as pd
        df = pd.read_csv(path)
        return cls.from_records(df.to_dict("records"), schema)

    @classmethod
    def from_jsonl(cls, path: str, schema: Schema) -> "Corpus":
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return cls.from_records(records, schema)


# ============================================================
# 3) Embedder
#
# Embedding abstraction layer.
# Any embedding backend can be swapped in by implementing
# the same interface.
# ============================================================

def _l2norm(x: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization.

    After normalization, inner product becomes equivalent
    to cosine similarity.
    """
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype("float32")


class Embedder(ABC):
    """
    Abstract embedding interface.
    """

    name: str

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        ...

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder(Embedder):
    """
    Dense retrieval backend powered by SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_instruction: str =
        "Represent this sentence for searching relevant passages: "
    ):
        self.name = model_name
        self.query_instruction = query_instruction
        self._model = None

    def _lazy(self):
        """
        Lazily load the model to reduce startup cost.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)

        return self._model

    def fit(self, texts):
        """
        Pretrained embedding models require no fitting.
        """
        pass

    def encode_documents(self, texts):
        return (
            self._lazy()
            .encode(
                list(texts),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            .astype("float32")
        )

    def encode_queries(self, queries):
        """
        BGE-style query instruction improves retrieval alignment.
        """
        q = [
            self.query_instruction + x
            for x in queries
        ]

        return (
            self._lazy()
            .encode(
                q,
                normalize_embeddings=True
            )
            .astype("float32")
        )


class TfidfEmbedder(Embedder):
    """
    Sparse lexical retrieval backend.

    Useful for baselines, offline demos,
    and small datasets.
    """

    def __init__(self, max_features: int = 20000):

        self.name = f"tfidf-{max_features}"

        from sklearn.feature_extraction.text import (
            TfidfVectorizer
        )

        self._vec = TfidfVectorizer(
            max_features=max_features,
            stop_words="english"
        )

        self._fitted = False

    def fit(self, texts):
        self._vec.fit(list(texts))
        self._fitted = True

    def _enc(self, texts):
        return _l2norm(
            self._vec
            .transform(list(texts))
            .toarray()
            .astype("float32")
        )

    def encode_documents(self, texts):

        if not self._fitted:
            self.fit(texts)

        return self._enc(texts)

    def encode_queries(self, queries):
        return self._enc(queries)


# ============================================================
# 4) VectorIndex
#
# Vector search abstraction.
# Uses FAISS when available and falls back to
# NumPy-based exhaustive search otherwise.
# ============================================================

class VectorIndex:

    def __init__(self, dim: int):

        self.dim = dim
        self._matrix = None

        try:
            import faiss

            # Exact inner-product search.
            # Large-scale deployments may replace this
            # with IVF or HNSW.
            self._faiss = faiss.IndexFlatIP(dim)

        except Exception:
            self._faiss = None

    def add(self, vectors: np.ndarray):

        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._matrix = vectors

    def search(self, q: np.ndarray, k: int):

        if self._faiss is not None:

            scores, idx = self._faiss.search(q, k)

            return scores[0], idx[0]

        # Inner product equals cosine similarity
        # after normalization.
        sims = self._matrix @ q[0]

        idx = np.argsort(sims)[::-1][:k]

        return sims[idx], idx


# ============================================================
# 5) Metadata Filtering
#
# Generic filtering independent of corpus type.
# ============================================================

_OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
    "in": lambda v, s: v in s,
    "contains": lambda v, s:
        isinstance(v, str)
        and str(s).lower() in v.lower(),
}


def match(meta: dict, filters: dict) -> bool:
    """
    Evaluate metadata filters against a document.
    """

    for key, cond in filters.items():

        value = meta.get(key)

        conditions = (
            cond
            if isinstance(cond, dict)
            else {"eq": cond}
        )

        for op, target in conditions.items():

            fn = _OPS.get(op)

            if fn is None:
                raise ValueError(
                    f"Unknown operator: {op}"
                )

            try:

                if value is None or not fn(
                    value,
                    target
                ):
                    return False

            except TypeError:
                # Type mismatches are treated as
                # filter failures.
                return False

    return True


# ============================================================
# 6) SemanticRetriever
#
# End-to-end retrieval engine.
# ============================================================

class SemanticRetriever:
    """
    Responsibilities:

        - document encoding
        - vector indexing
        - retrieval
        - metadata filtering
        - persistence
    """

    def __init__(
        self,
        schema: Schema,
        embedder: Embedder
    ):
        self.schema = schema
        self.embedder = embedder
        self.version = schema.version(
            embedder.name
        )

        self.corpus = None
        self.index = None
        self.embeddings = None

    def build(
        self,
        corpus: Corpus
    ) -> "SemanticRetriever":
        """
        Encode documents and build the retrieval index.
        """

        self.corpus = corpus

        self.embeddings = (
            self.embedder
            .encode_documents(corpus.texts)
        )

        self.index = VectorIndex(
            self.embeddings.shape[1]
        )

        self.index.add(
            self.embeddings
        )

        return self

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
        overfetch: int = 10,
    ) -> list[dict]:
        """
        Execute semantic retrieval with optional
        metadata filtering.
        """

        qv = self.embedder.encode_queries(
            [query]
        )

        # Over-fetch candidates before filtering
        # to avoid losing relevant results.
        n = min(
            (k * overfetch)
            if filters
            else k,
            len(self.corpus),
        )

        scores, idx = self.index.search(
            qv,
            n
        )

        hits = []

        for score, i in zip(scores, idx):

            i = int(i)

            meta = self.corpus.metadata[i]

            if filters and not match(
                meta,
                filters
            ):
                continue

            hits.append({
                "id": self.corpus.ids[i],
                "score": float(score),
                "text": self.corpus.texts[i],
                "metadata": meta,
            })

            if len(hits) >= k:
                break

        return hits

    # Persist embeddings, corpus data,
    # and version metadata.
    def save(self, path: str):

        import os

        os.makedirs(
            path,
            exist_ok=True
        )

        np.save(
            f"{path}/embeddings.npy",
            self.embeddings
        )

        with open(
            f"{path}/corpus.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "ids": self.corpus.ids,
                    "texts": self.corpus.texts,
                    "metadata": self.corpus.metadata,
                },
                f,
                ensure_ascii=False,
            )

        with open(
            f"{path}/meta.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "version": self.version,
                    "model": self.embedder.name,
                },
                f,
            )

    def load(
        self,
        path: str
    ) -> "SemanticRetriever":
        """
        Restore a previously built retrieval index.
        """

        self.embeddings = np.load(
            f"{path}/embeddings.npy"
        )

        with open(
            f"{path}/corpus.json",
            encoding="utf-8"
        ) as f:

            corpus = json.load(f)

        self.corpus = Corpus(
            corpus["ids"],
            corpus["texts"],
            corpus["metadata"],
        )

        # Rebuild TF-IDF vocabulary when needed.
        # SentenceTransformer ignores this call.
        self.embedder.fit(
            self.corpus.texts
        )

        self.index = VectorIndex(
            self.embeddings.shape[1]
        )

        self.index.add(
            self.embeddings
        )

        return self