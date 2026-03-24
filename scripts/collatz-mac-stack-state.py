#!/usr/bin/env python3
"""Internal helper for mac-dev-stack.sh: read/write .runtime/mac-dev-stack.json."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_write(args: argparse.Namespace) -> None:
    workers = json.loads(args.workers_json)
    payload = {
        "backend_pid": int(args.backend_pid),
        "dashboard_pid": int(args.dashboard_pid),
        "workers": workers,
        "health_url": args.health_url,
    }
    path = Path(args.state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cmd_print_pids(args: argparse.Namespace) -> None:
    """One line: backend|dashboard|worker_pid|worker_pid|... (for bash IFS='|')."""
    p = Path(args.state_path)
    if not p.is_file():
        print("")
        raise SystemExit(1)
    data = _load(p)
    bp = int(data.get("backend_pid") or 0)
    dp = int(data.get("dashboard_pid") or 0)
    workers = data.get("workers")
    if not workers and data.get("worker_pid"):
        workers = [{"name": "mac-managed-worker", "hardware": "auto", "pid": int(data["worker_pid"])}]
    if not workers:
        workers = []
    parts = [str(bp), str(dp)] + [str(int(w["pid"])) for w in workers if w.get("pid")]
    print("|".join(parts))


def cmd_print_status(args: argparse.Namespace) -> None:
    p = Path(args.state_path)
    if not p.is_file():
        return
    data = _load(p)
    bp = int(data.get("backend_pid") or 0)
    dp = int(data.get("dashboard_pid") or 0)
    workers = data.get("workers")
    if not workers and data.get("worker_pid"):
        workers = [{"name": "mac-managed-worker", "hardware": "auto", "pid": int(data["worker_pid"])}]
    workers = workers or []
    print(f"backend_pid={bp} alive={'yes' if _pid_alive(bp) else 'no'}")
    print(f"dashboard_pid={dp} alive={'yes' if _pid_alive(dp) else 'no'}")
    for i, w in enumerate(workers):
        pid = int(w.get("pid") or 0)
        name = w.get("name", "?")
        hw = w.get("hardware", "?")
        print(f"  worker[{i}] {name} hardware={hw} pid={pid} alive={'yes' if _pid_alive(pid) else 'no'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)

    w = sp.add_parser("write", help="Write stack state JSON")
    w.add_argument("state_path")
    w.add_argument("health_url")
    w.add_argument("backend_pid", type=int)
    w.add_argument("dashboard_pid", type=int)
    w.add_argument("workers_json")
    w.set_defaults(func=cmd_write)

    pp = sp.add_parser("print-pids", help="Print one line: backend|dashboard|w1|w2|...")
    pp.add_argument("state_path")
    pp.set_defaults(func=cmd_print_pids)

    ps = sp.add_parser("print-status", help="Human-readable status lines")
    ps.add_argument("state_path")
    ps.set_defaults(func=cmd_print_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
