from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from urllib import error, request

from .schemas import (
    ConsensusBaseline,
    LLMAutopilotRun,
    LLMAutopilotTask,
    LLMReviewDraft,
    MapVariant,
    ReviewRubric,
    Source,
    SourceStatus,
)


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    enabled: bool
    model: str
    api_key: str
    base_url: str
    safety_mode: str


@dataclass(frozen=True)
class LLMProviderStatus:
    provider: str
    enabled: bool
    configured: bool
    ready: bool
    model: str
    has_api_key: bool
    base_url: str | None
    safety_mode: str
    setup_required: bool
    autopilot_ready: bool
    memory_mode: str
    note: str


def _provider_config() -> LLMProviderConfig:
    provider = os.getenv("COLLATZ_LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    enabled = os.getenv("COLLATZ_LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    base_url = os.getenv(
        "GEMINI_API_BASE_URL",
        "https://generativelanguage.googleapis.com",
    ).strip() or "https://generativelanguage.googleapis.com"
    safety_mode = os.getenv("COLLATZ_LLM_SAFETY_MODE", "review_only").strip() or "review_only"
    return LLMProviderConfig(
        provider=provider,
        enabled=enabled,
        model=model,
        api_key=api_key,
        base_url=base_url,
        safety_mode=safety_mode,
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def get_llm_provider_status() -> dict[str, object]:
    config = _provider_config()
    if config.provider != "gemini":
        status = LLMProviderStatus(
            provider=config.provider,
            enabled=config.enabled,
            configured=False,
            ready=False,
            model=config.model,
            has_api_key=bool(config.api_key),
            base_url=config.base_url,
            safety_mode=config.safety_mode,
            setup_required=True,
            autopilot_ready=False,
            memory_mode="local_history",
            note="Only the Gemini scaffold is prepared right now.",
        )
        return asdict(status)

    configured = bool(config.api_key)
    ready = config.enabled and configured
    note = (
        "Gemini is configured for local-only assisted review and planning."
        if ready
        else "Gemini scaffold is present, but no local API key is enabled yet."
    )
    status = LLMProviderStatus(
        provider=config.provider,
        enabled=config.enabled,
        configured=configured,
        ready=ready,
        model=config.model,
        has_api_key=bool(config.api_key),
        base_url=config.base_url,
        safety_mode=config.safety_mode,
        setup_required=not ready,
        autopilot_ready=ready,
        memory_mode="local_history",
        note=note,
    )
    return asdict(status)


def _require_ready_config() -> LLMProviderConfig:
    config = _provider_config()
    if config.provider != "gemini":
        raise RuntimeError("Only the Gemini provider is supported right now.")
    if not config.enabled:
        raise RuntimeError("Gemini support is disabled. Set COLLATZ_LLM_ENABLED=1 locally.")
    if not config.api_key:
        raise RuntimeError("Gemini API key is missing from the local environment.")
    return config


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json_text(raw_text: str, *, error_label: str) -> dict[str, object]:
    candidate = _strip_fences(raw_text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(error_label)
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise RuntimeError(error_label) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(error_label)
    return parsed


def _enum_value(raw: object, *, allowed: set[str], fallback: str) -> str:
    value = str(raw or "").strip().lower()
    return value if value in allowed else fallback


def _bool_or_none(raw: object) -> bool | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"true", "yes", "1"}:
        return True
    if value in {"false", "no", "0"}:
        return False
    return None


def _string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text:
            values.append(text)
    return values


def _baseline_text(baseline: ConsensusBaseline) -> str:
    lines = [
        f"Problem status: {baseline.problem_status}",
        f"Checked as of: {baseline.checked_as_of}",
        f"Verified up to: {baseline.verified_up_to}",
        f"Note: {baseline.note}",
    ]
    for item in baseline.items:
        lines.append(f"- {item.title}: {item.detail} ({item.source_url})")
    return "\n".join(lines)


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_directions_snapshot(directions: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "slug": item.get("slug"),
            "status": item.get("status"),
            "score": item.get("score"),
            "description": _truncate(item.get("description"), 160),
            "success": _truncate(item.get("success_criteria"), 160),
            "abandon": _truncate(item.get("abandon_criteria"), 160),
        }
        for item in directions[:8]
    ]


def _compact_tasks_snapshot(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": item.get("id"),
            "direction_slug": item.get("direction_slug"),
            "title": _truncate(item.get("title"), 100),
            "kind": item.get("kind"),
            "priority": item.get("priority"),
            "status": item.get("status"),
        }
        for item in tasks[:12]
    ]


def _compact_claims_snapshot(claims: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": item.get("id"),
            "direction_slug": item.get("direction_slug"),
            "title": _truncate(item.get("title"), 100),
            "status": item.get("status"),
            "statement": _truncate(item.get("statement"), 220),
        }
        for item in claims[:10]
    ]


def _compact_sources_snapshot(sources: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": item.get("id"),
            "direction_slug": item.get("direction_slug"),
            "title": _truncate(item.get("title"), 100),
            "claim_type": item.get("claim_type"),
            "review_status": item.get("review_status"),
            "source_type": item.get("source_type"),
            "map_variant": item.get("map_variant"),
            "summary": _truncate(item.get("summary"), 180),
            "fallacy_tags": item.get("fallacy_tags", [])[:4],
        }
        for item in sources[:10]
    ]


def _compact_runs_snapshot(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    compacted: list[dict[str, object]] = []
    for item in runs[:12]:
        metrics = item.get("metrics") or {}
        compacted.append(
            {
                "id": item.get("id"),
                "direction_slug": item.get("direction_slug"),
                "status": item.get("status"),
                "kernel": item.get("kernel"),
                "hardware": item.get("hardware"),
                "range": [item.get("range_start"), item.get("range_end")],
                "summary": _truncate(item.get("summary"), 160),
                "processed": metrics.get("processed") or metrics.get("last_processed"),
                "max_total_stopping_time": metrics.get("max_total_stopping_time"),
                "max_stopping_time": metrics.get("max_stopping_time"),
                "max_excursion": metrics.get("max_excursion"),
            }
        )
    return compacted


def build_gemini_review_prompt(
    *,
    source: Source,
    baseline: ConsensusBaseline,
    allowed_tags: list[str],
) -> str:
    allowed_tags_block = ", ".join(sorted(set(allowed_tags)))
    return (
        "You are assisting a local Collatz research lab.\n"
        "Do not claim that the Collatz conjecture is solved.\n"
        "You are drafting a review suggestion only. You are not the source of truth.\n"
        "Return strict JSON only, with no markdown fences.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "summary": "one short paragraph",\n'
        '  "review_status": "intake|under_review|flagged|supported|refuted|context",\n'
        '  "map_variant": "unspecified|standard|shortcut|odd_only|inverse_tree",\n'
        '  "fallacy_tags": ["choose only from the allowed tag list"],\n'
        '  "rubric": {\n'
        '    "peer_reviewed": true|false|null,\n'
        '    "acknowledged_errors": true|false|null,\n'
        '    "defines_map_variant": true|false|null,\n'
        '    "distinguishes_empirical_from_proof": true|false|null,\n'
        '    "proves_descent": true|false|null,\n'
        '    "proves_cycle_exclusion": true|false|null,\n'
        '    "uses_statistical_argument": true|false|null,\n'
        '    "validation_backed": true|false|null,\n'
        '    "notes": "short rubric note"\n'
        "  },\n"
        '  "notes": "review notes for the lab operator",\n'
        '  "candidate_claims": ["bounded, testable claims only"],\n'
        '  "deterministic_checks": ["specific deterministic follow-up checks"],\n'
        '  "warnings": ["short caveats"]\n'
        "}\n\n"
        f"Allowed fallacy tags: {allowed_tags_block}\n\n"
        f"Consensus baseline:\n{_baseline_text(baseline)}\n\n"
        f"Source title: {source.title}\n"
        f"Source type: {source.source_type.value}\n"
        f"Claim type: {source.claim_type.value}\n"
        f"Current review status: {source.review_status.value}\n"
        f"Current map variant: {source.map_variant.value}\n"
        f"Authors: {source.authors or 'n/a'}\n"
        f"Year: {source.year or 'n/a'}\n"
        f"URL: {source.url or 'n/a'}\n"
        f"Summary: {source.summary or 'n/a'}\n"
        f"Notes: {source.notes or 'n/a'}\n"
        f"Existing fallacy tags: {', '.join(source.fallacy_tags) if source.fallacy_tags else 'none'}\n"
        f"Existing rubric: {json.dumps(source.rubric.model_dump(), ensure_ascii=True)}\n\n"
        "Be conservative. Prefer under_review or flagged unless the source is clearly just contextual background."
    )


def build_gemini_autopilot_prompt(
    *,
    baseline: ConsensusBaseline,
    directions: list[dict[str, object]],
    tasks: list[dict[str, object]],
    claims: list[dict[str, object]],
    sources: list[dict[str, object]],
    runs: list[dict[str, object]],
    max_tasks: int,
) -> str:
    direction_slugs = [str(item.get("slug", "")).strip() for item in directions if item.get("slug")]
    return (
        "You are assisting a local-first Collatz research lab.\n"
        "Do not claim that the Collatz conjecture is solved.\n"
        "You are not allowed to promote a proof, close the problem, or mark a source as valid truth.\n"
        "You are acting as a guarded planning assistant with local memory only.\n"
        "Return strict JSON only, with no markdown fences.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "summary": "one short paragraph",\n'
        '  "recommended_direction_slug": "one of the allowed direction slugs",\n'
        '  "recommended_direction_reason": "one short paragraph",\n'
        '  "notes": ["short planning notes"],\n'
        '  "task_proposals": [\n'
        "    {\n"
        '      "direction_slug": "one of the allowed direction slugs",\n'
        '      "title": "short task title",\n'
        '      "kind": "analysis|experiment|review|structure|source-review|theory",\n'
        '      "description": "clear deterministic task",\n'
        '      "priority": 1\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Allowed direction slugs: {', '.join(direction_slugs)}\n"
        f"Maximum task proposals: {max_tasks}\n\n"
        f"Consensus baseline:\n{_baseline_text(baseline)}\n\n"
        f"Directions snapshot:\n{json.dumps(_compact_directions_snapshot(directions), ensure_ascii=True)}\n\n"
        f"Open tasks snapshot:\n{json.dumps(_compact_tasks_snapshot(tasks), ensure_ascii=True)}\n\n"
        f"Claims snapshot:\n{json.dumps(_compact_claims_snapshot(claims), ensure_ascii=True)}\n\n"
        f"Source snapshot:\n{json.dumps(_compact_sources_snapshot(sources), ensure_ascii=True)}\n\n"
        f"Run snapshot:\n{json.dumps(_compact_runs_snapshot(runs), ensure_ascii=True)}\n\n"
        "Prefer work that is not just more brute-force verification.\n"
        "Bias toward structural analysis, claim hygiene, source review, modular falsification, inverse-tree work, and bounded next steps.\n"
        "Do not create duplicate tasks if the same work already exists.\n"
        "Do not propose more than one pure verification sweep unless no other direction is actionable.\n"
        "Treat the current database as the memory of the project."
        "\nDo not keep proposing more consolidation/admin tasks if the workspace is already full of them."
        "\nIf there is executable backlog already, prefer using it before inventing more tasks."
    )


def _request_gemini_json(prompt: str, config: LLMProviderConfig) -> dict[str, object]:
    url = f"{config.base_url.rstrip('/')}/v1beta/models/{config.model}:generateContent"
    generation_config: dict[str, object] = {
        "temperature": 0.1,
        "responseMimeType": "application/json",
        "maxOutputTokens": 1000,
    }
    model_name = config.model.strip().lower()
    if model_name.startswith("gemini-2.5"):
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}
    elif model_name.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": generation_config,
    }
    request_obj = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": config.api_key,
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with request.urlopen(request_obj, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network/runtime behavior
            body = exc.read().decode("utf-8", errors="replace")
            last_error = exc
            if exc.code in {429, 500, 503} and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            if exc.code == 429:
                raise RuntimeError(
                    f"Gemini request hit a rate or quota limit ({exc.code}). Reduce autopilot frequency, switch to a cheaper model, or wait for quota reset. Raw response: {body}"
                ) from exc
            raise RuntimeError(f"Gemini request failed ({exc.code}): {body}") from exc
        except error.URLError as exc:  # pragma: no cover - network/runtime behavior
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc
    raise RuntimeError(f"Gemini request failed after retries: {last_error}")


def _extract_gemini_text(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    first = candidates[0]
    if not isinstance(first, dict):
        raise RuntimeError("Gemini returned an invalid candidate payload.")
    content = first.get("content")
    if not isinstance(content, dict):
        raise RuntimeError("Gemini returned no content payload.")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError("Gemini returned no content parts.")
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    text = "\n".join(chunk for chunk in chunks if chunk.strip()).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty review draft.")
    return text


def generate_source_review_draft(
    *,
    source: Source,
    baseline: ConsensusBaseline,
    allowed_tags: list[str],
) -> LLMReviewDraft:
    config = _require_ready_config()
    prompt = build_gemini_review_prompt(
        source=source,
        baseline=baseline,
        allowed_tags=allowed_tags,
    )
    raw_payload = _request_gemini_json(prompt, config)
    raw_text = _extract_gemini_text(raw_payload)
    parsed = _parse_json_text(raw_text, error_label="Gemini returned a non-JSON review draft.")

    warnings = _string_list(parsed.get("warnings"))
    allowed_tag_set = set(allowed_tags)
    normalized_tags: list[str] = []
    unknown_tags: list[str] = []
    for tag in _string_list(parsed.get("fallacy_tags")):
        lowered = tag.strip().lower()
        if lowered in allowed_tag_set and lowered not in normalized_tags:
            normalized_tags.append(lowered)
        elif lowered and lowered not in unknown_tags:
            unknown_tags.append(lowered)
    if unknown_tags:
        warnings.append(f"Ignored unknown fallacy tags: {', '.join(unknown_tags)}")

    rubric_raw = parsed.get("rubric")
    rubric_data = rubric_raw if isinstance(rubric_raw, dict) else {}
    rubric = ReviewRubric(
        peer_reviewed=_bool_or_none(rubric_data.get("peer_reviewed")),
        acknowledged_errors=_bool_or_none(rubric_data.get("acknowledged_errors")),
        defines_map_variant=_bool_or_none(rubric_data.get("defines_map_variant")),
        distinguishes_empirical_from_proof=_bool_or_none(
            rubric_data.get("distinguishes_empirical_from_proof")
        ),
        proves_descent=_bool_or_none(rubric_data.get("proves_descent")),
        proves_cycle_exclusion=_bool_or_none(rubric_data.get("proves_cycle_exclusion")),
        uses_statistical_argument=_bool_or_none(rubric_data.get("uses_statistical_argument")),
        validation_backed=_bool_or_none(rubric_data.get("validation_backed")),
        notes=str(rubric_data.get("notes") or "").strip(),
    )

    review_status = SourceStatus(
        _enum_value(
            parsed.get("review_status"),
            allowed={item.value for item in SourceStatus},
            fallback=SourceStatus.UNDER_REVIEW.value,
        )
    )
    map_variant = MapVariant(
        _enum_value(
            parsed.get("map_variant"),
            allowed={item.value for item in MapVariant},
            fallback=source.map_variant.value if source.map_variant else MapVariant.UNSPECIFIED.value,
        )
    )

    return LLMReviewDraft(
        source_id=source.id,
        provider=config.provider,
        model=config.model,
        safety_mode=config.safety_mode,
        created_at=_utc_now(),
        review_status=review_status,
        map_variant=map_variant,
        summary=str(parsed.get("summary") or "").strip(),
        notes=str(parsed.get("notes") or "").strip(),
        fallacy_tags=normalized_tags,
        rubric=rubric,
        candidate_claims=_string_list(parsed.get("candidate_claims")),
        deterministic_checks=_string_list(parsed.get("deterministic_checks")),
        warnings=warnings,
        raw_text=_strip_fences(raw_text),
    )


def generate_autopilot_plan(
    *,
    baseline: ConsensusBaseline,
    directions: list[dict[str, object]],
    tasks: list[dict[str, object]],
    claims: list[dict[str, object]],
    sources: list[dict[str, object]],
    runs: list[dict[str, object]],
    max_tasks: int = 3,
) -> LLMAutopilotRun:
    config = _require_ready_config()
    prompt = build_gemini_autopilot_prompt(
        baseline=baseline,
        directions=directions,
        tasks=tasks,
        claims=claims,
        sources=sources,
        runs=runs,
        max_tasks=max_tasks,
    )
    raw_payload = _request_gemini_json(prompt, config)
    raw_text = _extract_gemini_text(raw_payload)
    parsed = _parse_json_text(raw_text, error_label="Gemini returned a non-JSON autopilot plan.")

    allowed_directions = {str(item.get("slug", "")).strip() for item in directions if item.get("slug")}
    task_proposals: list[LLMAutopilotTask] = []
    for raw_task in parsed.get("task_proposals", []) if isinstance(parsed.get("task_proposals"), list) else []:
        if not isinstance(raw_task, dict):
            continue
        direction_slug = str(raw_task.get("direction_slug") or "").strip()
        if direction_slug not in allowed_directions:
            continue
        title = str(raw_task.get("title") or "").strip()
        description = str(raw_task.get("description") or "").strip()
        kind = str(raw_task.get("kind") or "analysis").strip().lower() or "analysis"
        try:
            priority = int(raw_task.get("priority") or 2)
        except (TypeError, ValueError):
            priority = 2
        if not title or not description:
            continue
        task_proposals.append(
            LLMAutopilotTask(
                direction_slug=direction_slug,
                title=title[:140],
                kind=kind[:40],
                description=description[:1200],
                priority=max(1, min(5, priority)),
            )
        )
        if len(task_proposals) >= max_tasks:
            break

    recommended_direction_slug = str(parsed.get("recommended_direction_slug") or "").strip()
    if recommended_direction_slug not in allowed_directions:
        recommended_direction_slug = task_proposals[0].direction_slug if task_proposals else ""

    return LLMAutopilotRun(
        provider=config.provider,
        model=config.model,
        safety_mode=config.safety_mode,
        created_at=_utc_now(),
        summary=str(parsed.get("summary") or "").strip(),
        recommended_direction_slug=recommended_direction_slug,
        recommended_direction_reason=str(parsed.get("recommended_direction_reason") or "").strip(),
        memory_mode="local_history",
        notes=_string_list(parsed.get("notes")),
        task_proposals=task_proposals,
        applied=False,
        applied_task_ids=[],
    )
