from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


_CACHE_TTL_SECONDS = 90
_cache_lock = Lock()
_cache: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}
_TRACKED_COMMENT_SPECS = (
    {
        "signal": "structure",
        "title": "Need deeper structure-analysis tools",
        "url": "https://www.reddit.com/r/Collatz/comments/1s02k3b/comment/obqz759/",
        "takeaway": "The app is already useful as a journal and validator, but it needs tools for structural analysis beyond orbit playback, stopping time, and excursion.",
        "implemented_note": "This feedback directly pushed structure-analysis tasks into the backlog instead of keeping the app orbit-centric.",
        "implemented_items": [
            {"id": "COL-0088", "label": "Parity-vector filter framework"},
            {"id": "COL-0098", "label": "Investigate parity-vector filters with new record-breakers"},
            {"id": "COL-0106", "label": "Develop new parity-vector filters from consolidated record data"},
        ],
    },
    {
        "signal": "indirect",
        "title": "Keep a lane for indirect transforms",
        "url": "https://www.reddit.com/r/Collatz/comments/1s02k3b/comment/obqzo23/",
        "takeaway": "Indirect approaches can be valuable, but they should be treated as explicit theory tasks with preserved-property and proof-obligation tracking.",
        "implemented_note": "This feedback was converted into a concrete theory task rather than left as vague inspiration.",
        "implemented_items": [
            {"id": "COL-0091", "label": "Catalog indirect transforms and surrogate maps"},
        ],
    },
)


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


def _fetch_listing(url: str) -> Any:
    request = Request(
        url,
        headers={"User-Agent": "CollatzLab/0.1 (local research dashboard)"},
    )
    with urlopen(request, timeout=8) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


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


def _comment_json_url(url: str) -> str:
    normalized = (url or "").strip()
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    path = parsed.path.rstrip("/")
    return f"https://www.reddit.com{path}/.json?raw_json=1"


def _extract_comment_data(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    listing = payload[1]
    if not isinstance(listing, dict):
        return None
    children = listing.get("data", {}).get("children", [])
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        if isinstance(data, dict):
            return data
    return None


def _fetch_tracked_comments() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for spec in _TRACKED_COMMENT_SPECS:
        try:
            payload = _fetch_listing(_comment_json_url(spec["url"]))
            data = _extract_comment_data(payload)
            if not data:
                continue
            items.append(
                {
                    "id": str(data.get("id") or ""),
                    "author": str(data.get("author") or "unknown"),
                    "permalink": f"https://www.reddit.com{data.get('permalink') or ''}",
                    "created_at": _iso_from_utc(data.get("created_utc")),
                    "score": int(data.get("score") or 0),
                    "signal": spec["signal"],
                    "title": spec["title"],
                    "body": _truncate(" ".join(str(data.get("body") or "").split()), 320),
                    "takeaway": spec["takeaway"],
                    "implemented_note": spec["implemented_note"],
                    "implemented_items": list(spec["implemented_items"]),
                }
            )
        except Exception:
            continue
    return items


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
        "tracked_comments": _fetch_tracked_comments(),
    }
    with _cache_lock:
        _cache[cache_key] = (time.time(), result)
    return result
