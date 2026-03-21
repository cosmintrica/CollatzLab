from __future__ import annotations

import os
from dataclasses import dataclass, asdict


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
    note: str


def get_llm_provider_status() -> dict[str, object]:
    provider = os.getenv("COLLATZ_LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    enabled = os.getenv("COLLATZ_LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    base_url = os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com")
    safety_mode = os.getenv("COLLATZ_LLM_SAFETY_MODE", "review_only")

    if provider != "gemini":
        status = LLMProviderStatus(
            provider=provider,
            enabled=enabled,
            configured=False,
            ready=False,
            model=model,
            has_api_key=bool(api_key),
            base_url=base_url,
            safety_mode=safety_mode,
            note="Only the Gemini scaffold is prepared right now.",
        )
        return asdict(status)

    configured = bool(api_key)
    ready = enabled and configured
    note = (
        "Gemini is configured for local-only assisted review and planning."
        if ready
        else "Gemini scaffold is present, but no local API key is enabled yet."
    )
    status = LLMProviderStatus(
        provider=provider,
        enabled=enabled,
        configured=configured,
        ready=ready,
        model=model,
        has_api_key=bool(api_key),
        base_url=base_url,
        safety_mode=safety_mode,
        note=note,
    )
    return asdict(status)


def build_gemini_review_prompt(*, source_title: str, source_summary: str, baseline: str) -> str:
    return (
        "You are assisting a local Collatz research lab. "
        "Do not claim the conjecture is solved. "
        "Extract candidate lemmas, red flags, and deterministic follow-up checks.\n\n"
        f"Source title: {source_title}\n"
        f"Source summary: {source_summary}\n"
        f"Consensus baseline: {baseline}\n\n"
        "Return only: 1) candidate claims 2) likely fallacy tags 3) modular probes to run."
    )


def build_gemini_experiment_prompt(*, direction: str, recent_findings: str) -> str:
    return (
        "You are proposing the next bounded Collatz experiment for a deterministic lab. "
        "Do not output proofs or certainty. "
        "Prefer falsifiable claims, residue filters, inverse-tree structure, or source-review tasks.\n\n"
        f"Direction: {direction}\n"
        f"Recent findings: {recent_findings}\n\n"
        "Return only: title, rationale, exact check, and failure condition."
    )
