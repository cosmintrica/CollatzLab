from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_CACHE_TTL_SECONDS = 90
_cache_lock = Lock()
_cache: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": "CollatzLab/0.1 (local research dashboard)"},
    )
    with urlopen(request, timeout=8) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Unexpected Reddit payload shape.")
    return data


def _iso_from_utc(timestamp: float | int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(float(timestamp), UTC).isoformat()


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _excerpt(post: dict[str, Any], limit: int = 180) -> str:
    body = str(post.get("selftext") or "").strip()
    if body:
        compact = " ".join(body.split())
        return _truncate(compact, limit)
    outbound = str(post.get("url_overridden_by_dest") or post.get("url") or "").strip()
    if outbound and not outbound.startswith("https://www.reddit.com/"):
        return _truncate(outbound, limit)
    return "No self-text preview."


def _review_signal(post: dict[str, Any]) -> str:
    title = str(post.get("title") or "").lower()
    body = str(post.get("selftext") or "").lower()
    haystack = f"{title} {body}"
    if any(token in haystack for token in ["tool", "app", "dashboard", "lab", "software"]):
        return "tooling"
    if any(token in title for token in ["solved", "proof", "proved", "prove", "lean4", "riemann", "grh"]):
        return "review"
    if any(token in body for token in ["i solved", "this proves", "full proof", "proved in lean", "proven in lean"]):
        return "review"
    return "watch"


def fetch_subreddit_feed(subreddit: str = "Collatz", sort: str = "new", limit: int = 8) -> dict[str, Any]:
    normalized_subreddit = (subreddit or "Collatz").strip() or "Collatz"
    normalized_sort = (sort or "new").strip().lower() or "new"
    normalized_limit = max(1, min(int(limit), 20))
    cache_key = (normalized_subreddit.lower(), normalized_sort, normalized_limit)
    now = time.time()

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

    query = urlencode({"limit": normalized_limit, "raw_json": 1})
    url = f"https://www.reddit.com/r/{normalized_subreddit}/{normalized_sort}.json?{query}"
    payload = _fetch_json(url)
    children = payload.get("data", {}).get("children", [])
    posts: list[dict[str, Any]] = []
    review_candidates = 0

    for child in children:
        if not isinstance(child, dict):
            continue
        data = child.get("data", {})
        if not isinstance(data, dict):
            continue
        signal = _review_signal(data)
        if signal == "review":
            review_candidates += 1
        permalink = str(data.get("permalink") or "")
        posts.append(
            {
                "id": str(data.get("id") or ""),
                "title": str(data.get("title") or "Untitled"),
                "author": str(data.get("author") or "unknown"),
                "permalink": f"https://www.reddit.com{permalink}" if permalink else "",
                "created_at": _iso_from_utc(data.get("created_utc")),
                "score": int(data.get("score") or 0),
                "num_comments": int(data.get("num_comments") or 0),
                "flair_text": str(data.get("link_flair_text") or ""),
                "signal": signal,
                "excerpt": _excerpt(data),
            }
        )

    result = {
        "subreddit": normalized_subreddit,
        "sort": normalized_sort,
        "fetched_at": datetime.now(UTC).isoformat(),
        "review_candidate_count": review_candidates,
        "posts": posts,
    }
    with _cache_lock:
        _cache[cache_key] = (time.time(), result)
    return result
