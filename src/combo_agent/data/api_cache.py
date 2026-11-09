"""A JSON-over-HTTP client that stores every response on disk, so builds can be replayed offline.

Each GET is cached as one file named by the sha256 of its URL, holding the URL, the HTTP status,
and the parsed body. A 404 is cached too (as `body: null`): "this name is not in ChEMBL" is a
result, and the build must see the same answer when it is replayed.

Offline mode (`offline=True`, or `COMBO_OFFLINE=1`) raises `CacheMiss` instead of touching the
network. That is how reproducibility is checked: a second build from the cache alone must give
byte-identical outputs.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

USER_AGENT = "combo-agent-data-harmonize/0.1"


class CacheMiss(LookupError):
    """A URL was requested in offline mode and is not in the cache."""


class ApiError(OSError):
    """A request failed after all retries (not a 404)."""


Fetcher = Callable[[str], tuple[int, Any]]
"""Takes a URL, returns (HTTP status, parsed JSON body or None)."""


def urllib_fetch(url: str, timeout: float = 60.0) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # 404 = not found; 400 = PubChem's answer to a name it cannot parse. Both are permanent
        # answers about this URL, so they are cached as "no result" rather than retried.
        if e.code in (400, 404):
            return e.code, None
        raise


class CachedJsonClient:
    """GET JSON with an on-disk cache, retries, and an optional per-host minimum interval."""

    def __init__(
        self,
        cache_dir: Path,
        offline: bool | None = None,
        fetch: Fetcher = urllib_fetch,
        attempts: int = 8,
        min_interval: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache_dir = cache_dir
        self.offline = os.environ.get("COMBO_OFFLINE") == "1" if offline is None else offline
        self._fetch, self._attempts, self._sleep = fetch, attempts, sleep
        self._min_interval, self._last = min_interval, 0.0
        self.hits = self.misses = 0

    def _path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def get(self, url: str) -> Any:
        """Parsed JSON body for `url`, or None if the server answered 404 (or 400)."""
        path = self._path(url)
        if path.exists():
            entry = json.loads(path.read_text())
            if entry["url"] != url:  # a sha256 collision would be news; fail rather than guess
                raise RuntimeError(f"cache file {path} holds {entry['url']}, not {url}")
            self.hits += 1
            return entry["body"]
        if self.offline:
            raise CacheMiss(url)
        status, body = self._fetch_with_retries(url)
        self.misses += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps({"url": url, "status": status, "body": body}, sort_keys=True))
        tmp.replace(path)
        return body

    def _fetch_with_retries(self, url: str) -> tuple[int, Any]:
        last: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            try:
                return self._fetch(url)
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as e:
                # HTTPError (a URLError) covers 5xx and 429; 400/404 never reach here.
                # HTTPException covers IncompleteRead: a body cut short (seen on ChEMBL, 2026-11-08).
                last = e
                if attempt < self._attempts:
                    # Seen on ChEMBL 2026-11-08: a slow query answered with a transient 500 and
                    # then served, and large bodies (~180 kB) cut short about half the time. So
                    # retry often and back off generously: 5, 10, 20, 40, 60, 60, ... s.
                    self._sleep(min(5.0 * 2 ** (attempt - 1), 60.0))
        raise ApiError(f"GET {url} failed after {self._attempts} attempts: {last}") from last
