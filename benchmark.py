"""
benchmark.py

Lightweight load-testing utility for the search service.

Measures:

- Throughput (QPS)
- P50 latency
- P95 latency
- P99 latency
- Cache hit rate

Designed to evaluate the impact of caching on both
average latency and tail latency.

Usage:

    python benchmark.py

    python benchmark.py \
        http://127.0.0.1:8000 \
        800 \
        32

Arguments:
    URL              Search service endpoint
    Total Requests   Number of requests to issue
    Concurrency      Number of concurrent workers

Start the search service before running this benchmark.
"""

import sys
import time
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 800
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 32

# Use a small set of repeated queries intentionally
# to generate cache hits and demonstrate the impact
# of caching on latency and throughput.
QUERIES = [
    "quiet family home near the park",
    "stylish downtown studio close to restaurants",
    "cozy private room on a budget",
    "bright apartment with a view and parking",
    "modern loft walking distance to transit",
]


def hit(q):
    """
    Execute a single search request and measure
    end-to-end response latency.
    """
    url = BASE + "/search?" + urllib.parse.urlencode({"q": q, "k": 10})
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    ms = (time.perf_counter() - t0) * 1000
    return ms, data.get("cached", False)


def pct(xs, p):
    """
    Compute the p-th percentile latency.
    """
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def run(label):
    # Generate a benchmark workload by repeatedly
    # cycling through a small query set.
    reqs = [QUERIES[i % len(QUERIES)] for i in range(TOTAL)]

    lat, cached_flags = [], []

    # Measure end-to-end benchmark duration.
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for ms, c in ex.map(hit, reqs):
            lat.append(ms)
            cached_flags.append(c)

    wall = time.perf_counter() - t0

    # Observed cache hit ratio during execution.
    hit_rate = sum(cached_flags) / len(cached_flags)

    print(f"\n[{label}]  {TOTAL} requests / concurrency {CONC}")
    print(f"  QPS         {TOTAL / wall:8.1f}")
    print(f"  P50         {pct(lat, 50):8.2f} ms")
    print(f"  P95         {pct(lat, 95):8.2f} ms")
    print(f"  P99         {pct(lat, 99):8.2f} ms")
    print(f"  Cache Hit   {hit_rate * 100:7.1f}%")


if __name__ == "__main__":

    # Benchmark steady-state performance after cache warm-up.
    # Repeated queries are used intentionally to evaluate
    # cache effectiveness and tail-latency reduction.
    run("cache-enabled workload")