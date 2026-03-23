# Incident Log

Critical bugs, outages, and data integrity issues. Every incident must have: root cause, impact, fix, and prevention measures.

---

## INC-001: Worker crash loop — `No module named 'src'` (2026-03-23)

**Severity:** High — silent data loss, verification gap, 71 spam failed runs

**Timeline:**
- 03:38-03:40 UTC — CPU worker COL-0046 crashes repeatedly (~1/second for 2 minutes)
- 03:40 UTC — Worker stops (no more queued runs to claim)
- ~04:00 UTC — Issue discovered during investigation

**Symptoms:**
- 71 failed runs with identical error: `Worker COL-0046 failed: No module named 'src'`
- COL-9532 partially completed (31B of 48B seeds verified) before crash
- 70 retries on next range all failed instantly
- No traceback captured — only the exception message string was stored

**Root cause: UNCONFIRMED**
No traceback was captured at the time of failure. The original error message (`No module named 'src'`) has no matching `from src...` or `import src` anywhere in the codebase. Hypotheses:
- (A) Python files edited mid-save while worker was running → transient `ModuleNotFoundError`. Plausible but unproven — many variables changed during debugging (taskkill, mixed Git Bash / PowerShell runs, stale worker IDs).
- (B) Numba cache from a prior layout referencing `src.collatz_lab...`. Checked `%LOCALAPPDATA%` and project `.nbi` files — no evidence found, but not conclusive.
- (C) Environmental difference in `Start-Process` vs direct shell. Tested post-fix and could not reproduce, but the test was not clean (too many variables changed).

**Status:** The error has not recurred since fixes were applied. If it returns, the traceback logging (now in place) will identify the exact import chain.

**Impact:**
- 17B seed verification gap: range 13,437,637,774,921 - 13,454,637,774,920
- 71 junk failed run records polluting the database
- No traceback = hours spent diagnosing what should have been obvious
- Spam loop: worker retried ~1/sec with no backoff, creating 70 identical failed runs in 2 minutes

**Fix (operational):**
1. Recovered 31B verified seeds from COL-9532 checkpoint → COL-9619 (completed)
2. Queued recovery tail → COL-9620 (completed after fix)
3. Deleted 70+61 duplicate spam failures
4. Added `PYTHONUNBUFFERED=1` to `run-worker.ps1` (visibility fix, not root cause fix)

**Prevention (implemented):**
- [x] Full traceback stored in run summary on failure (last 400 chars) + `logger.error()` with complete trace
- [x] File-based logging: `data/logs/worker-{name}.log` with timestamps
- [x] Retry backoff: exponential 10s→20s→40s→80s→160s on non-overflow failures
- [x] Max consecutive failure limit: worker stops after 5 consecutive failures with status `error` (`WorkerStatus.ERROR` added to schema)
- [x] `PYTHONUNBUFFERED=1` for visible worker output in redirect logs

**Lessons:**
- Never diagnose without a traceback. The first prevention measure for any error handler is `traceback.format_exc()`.
- `PYTHONUNBUFFERED` fixes log visibility, not the underlying error — don't confuse the two.
- One launch path for production: `run-worker.ps1` / `dev-stack.ps1` with canonical `PYTHONPATH`. Avoid ad-hoc `from src.collatz_lab...` patterns.

---

<!-- Template for new incidents:

## INC-XXX: Title (YYYY-MM-DD)

**Severity:** Critical / High / Medium

**Timeline:**
- HH:MM UTC — what happened

**Symptoms:**
- What was observed

**Root cause:**
Why it happened

**Impact:**
- Data loss, downtime, incorrect results

**Fix:**
1. What was done to resolve

**Prevention needed:**
- [ ] What should be done to prevent recurrence

-->
