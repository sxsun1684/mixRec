"""
service.py

FastAPI service layer for the retrieval system.

Features:
    /search
        Semantic retrieval with metadata filtering
        and response caching.

    /health
        Service health check endpoint.

    /version
        Active index version and cache statistics.

    /reload
        Hot-swap retrieval indexes without restarting
        the service.

The service defaults to a TF-IDF retriever, allowing
the entire API stack to run without external model
downloads.

Set EMBEDDER=bge to enable semantic retrieval with
Sentence Transformers.

Startup:

    pip install fastapi uvicorn

    EMBEDDER=tfidf uvicorn service:app --port 8000

    # or

    EMBEDDER=bge uvicorn service:app --port 8000
"""

import os
import json
import time
import math
import threading
from typing import Optional

from fastapi import FastAPI, Query

from retrieval import Schema, FieldSpec, Corpus, SemanticRetriever, TfidfEmbedder
from cache import TTLCache

# Service configuration (overridable via environment variables).
CSV = os.environ.get("LISTINGS_CSV", "listings_prepared.csv")
EMBEDDER = os.environ.get("EMBEDDER", "tfidf")          # "tfidf" | "bge"
CACHE_TTL = float(os.environ.get("CACHE_TTL", "300"))
CACHE_MAX = int(os.environ.get("CACHE_MAX", "2000"))

SCHEMA = Schema(
    id_field="id",
    text_spec=[FieldSpec("embed_text")],
    metadata_fields=["price_num", "room_type", "neighbourhood_cleansed"],
)

def build_embedder(name):
    """
    Create an embedding backend from configuration.

    Supports both lexical (TF-IDF) and semantic
    (Sentence Transformer) retrieval pipelines.
    """
    if name == "bge":
        from retrieval import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()

    return TfidfEmbedder(max_features=4000)


def to_native(v):
    """
    Convert NumPy scalars and NaN values into
    JSON-serializable native Python types.
    """
    if hasattr(v, "item"):
        v = v.item()

    if isinstance(v, float) and math.isnan(v):
        return None

    return v


# Retrieval engine state with version tracking
# and atomic index replacement.
class EngineState:
    """
    Manages the active retrieval index and enables
    atomic index reloads without service downtime.
    """

    def __init__(self):
        self.engine = None
        self.version = None
        self._lock = threading.Lock()

    def load(self, embedder_name):
        eng = SemanticRetriever(SCHEMA, build_embedder(embedder_name))
        eng.build(Corpus.from_csv(CSV, SCHEMA))

        version = f"{embedder_name}-{eng.version}"

        # Atomically replace the active index so
        # in-flight requests continue uninterrupted.
        with self._lock:
            self.engine = eng
            self.version = version

        return version

    def search(self, *a, **kw):
        return self.engine.search(*a, **kw)


STATE = EngineState()
CACHE = TTLCache(maxsize=CACHE_MAX, ttl=CACHE_TTL)

app = FastAPI(title="Semantic Search API")


def make_filters(max_price, room_type, neighbourhood):
    """
    Construct metadata filters from request parameters.
    """
    f = {}

    if max_price is not None:
        f["price_num"] = {"lte": max_price}

    if room_type:
        f["room_type"] = {"eq": room_type}

    if neighbourhood:
        f["neighbourhood_cleansed"] = {"eq": neighbourhood}

    return f

@app.get("/health")
def health():
    return {"status": "ok", "version": STATE.version}


@app.get("/version")
def version():
    return {"active_version": STATE.version, "cache": CACHE.stats()}


@app.get("/search")
def search(q: str,
           k: int = 10,
           max_price: Optional[float] = None,
           room_type: Optional[str] = None,
           neighbourhood: Optional[str] = None):

    filters = make_filters(max_price, room_type, neighbourhood)

    # Include the active index version in the cache key
    # so cached results are automatically invalidated
    # after an index reload.
    key = json.dumps(
        {"q": q, "k": k, "f": filters, "v": STATE.version},
        sort_keys=True,
    )

    t0 = time.perf_counter()

    cached = CACHE.get(key)

    if cached is not None:
        out = dict(cached)
        out["cached"] = True
        out["latency_ms"] = round(
            (time.perf_counter() - t0) * 1000,
            2,
        )
        return out

    hits = STATE.search(q, k=k, filters=filters)

    results = [{
        "id": to_native(h["id"]),
        "score": round(h["score"], 4),
        "title": h["text"].split("\n")[0].replace("Title: ", ""),
        "metadata": {
            kk: to_native(vv)
            for kk, vv in h["metadata"].items()
        },
    } for h in hits]

    resp = {
        "query": q,
        "version": STATE.version,
        "cached": False,
        "results": results,
    }

    CACHE.set(key, resp)

    out = dict(resp)

    out["latency_ms"] = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    return out


@app.post("/reload")
def reload(embedder: Optional[str] = Query(None)):
    """
    Hot-swap the active retrieval index.

    Rolling back is achieved by reloading a
    previous index version.
    """
    prev = STATE.version
    new = STATE.load(embedder or EMBEDDER)

    # Clear stale cache entries after
    # switching index versions.
    CACHE.clear()

    return {
        "previous": prev,
        "active": new,
    }


# Build the initial retrieval index during
# application startup.
STATE.load(EMBEDDER)