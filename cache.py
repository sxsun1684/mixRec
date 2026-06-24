"""
cache.py

Thread-safe in-memory cache with:

- TTL (time-to-live) expiration
- LRU (least recently used) eviction
- Cache hit/miss metrics

Why use an in-memory cache instead of Redis?

- Single-process deployments:
  avoids network round trips and external dependencies,
  providing the lowest possible latency.

- Multi-process or distributed deployments:
  can be replaced with Redis while preserving the same
  cache interface.

Caches retrieval results keyed by:

    (query, k, filters, version)

Repeated requests can be served directly from cache,
avoiding expensive embedding generation and retrieval.
"""

import time
import threading
from collections import OrderedDict


class TTLCache:
    """
    Thread-safe TTL + LRU cache.

    Entries expire after `ttl` seconds and are evicted
    using an LRU policy when capacity is exceeded.
    """

    def __init__(self, maxsize=1000, ttl=300.0):
        self.maxsize = maxsize
        self.ttl = ttl

        # key -> (timestamp, value)
        self._store = OrderedDict()

        # Synchronizes concurrent access.
        self._lock = threading.Lock()

        # Cache metrics.
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """
        Retrieve a value from cache.

        Returns None on cache miss or expiration.
        """
        with self._lock:

            item = self._store.get(key)

            if item is None:
                self.misses += 1
                return None

            ts, val = item

            # Remove expired entries.
            if time.time() - ts > self.ttl:
                del self._store[key]
                self.misses += 1
                return None

            # Mark as recently used.
            self._store.move_to_end(key)

            self.hits += 1
            return val

    def set(self, key, val):
        """
        Insert or update a cache entry.
        """
        with self._lock:

            self._store[key] = (time.time(), val)

            # Mark as most recently used.
            self._store.move_to_end(key)

            # Evict least recently used entries
            # when capacity is exceeded.
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self):
        """
        Remove all cached entries.
        """
        with self._lock:
            self._store.clear()

    def stats(self):
        """
        Return cache utilization metrics.
        """
        total = self.hits + self.misses

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                round(self.hits / total, 3)
                if total else 0.0
            ),
            "size": len(self._store),
        }