from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import Settings
from .hardware import discover_hardware, validate_execution_request
from .logutil import silence_numba_cuda_info
from .repository import LabRepository
from .services import execute_run, generate_report, validate_run
from .worker import start_worker_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab", description="Collatz Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the lab workspace")

    task_parser = subparsers.add_parser("task", help="Task commands")
    task_sub = task_parser.add_subparsers(dest="task_command", required=True)
    task_new = task_sub.add_parser("new", help="Create a task")
    task_new.add_argument("--direction", required=True)
    task_new.add_argument("--title", required=True)
    task_new.add_argument("--kind", required=True)
    task_new.add_argument("--description", required=True)
    task_new.add_argument("--owner", default="integrator")
    task_new.add_argument("--priority", type=int, default=2)

    run_parser = subparsers.add_parser("run", help="Run commands")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)
    run_start = run_sub.add_parser("start", help="Create and execute a run")
    run_start.add_argument("--direction", required=True)
    run_start.add_argument("--name", required=True)
    run_start.add_argument("--start", type=int, required=True)
    run_start.add_argument("--end", type=int, required=True)
    run_start.add_argument("--kernel", default="cpu-direct")
    run_start.add_argument("--owner", default="compute-agent")
    run_start.add_argument("--hardware", default="cpu")
    run_start.add_argument("--checkpoint-interval", type=int, default=250)
    run_start.add_argument("--enqueue-only", action="store_true")
    run_resume = run_sub.add_parser("resume", help="Resume an existing run")
    run_resume.add_argument("run_id")
    run_resume.add_argument("--checkpoint-interval", type=int, default=250)

    validate_parser = subparsers.add_parser("validate", help="Validate a run")
    validate_parser.add_argument("run_id")

    claim_parser = subparsers.add_parser("claim", help="Claim commands")
    claim_sub = claim_parser.add_subparsers(dest="claim_command", required=True)
    claim_new = claim_sub.add_parser("new", help="Create a claim")
    claim_new.add_argument("--direction", required=True)
    claim_new.add_argument("--title", required=True)
    claim_new.add_argument("--statement", required=True)
    claim_new.add_argument("--owner", default="theory-agent")
    claim_new.add_argument("--dependencies", default="")
    claim_new.add_argument("--notes", default="")
    claim_link = claim_sub.add_parser("link-run", help="Link a claim and run")
    claim_link.add_argument("claim_id")
    claim_link.add_argument("run_id")
    claim_link.add_argument("--relation", required=True, choices=["supports", "refutes", "relates"])

    direction_parser = subparsers.add_parser("direction", help="Direction commands")
    direction_sub = direction_parser.add_subparsers(dest="direction_command", required=True)
    direction_review = direction_sub.add_parser("review", help="Review a direction")
    direction_review.add_argument("slug")

    report_parser = subparsers.add_parser("report", help="Report commands")
    report_sub = report_parser.add_subparsers(dest="report_command", required=True)
    report_sub.add_parser("generate", help="Generate a Markdown report")

    worker_parser = subparsers.add_parser("worker", help="Worker commands")
    worker_sub = worker_parser.add_subparsers(dest="worker_command", required=True)
    worker_sub.add_parser("capabilities", help="Inspect local execution capabilities")

    worker_common = argparse.ArgumentParser(add_help=False)
    worker_common.add_argument("--name", default="local-worker")
    worker_common.add_argument("--role", default="compute-agent")
    worker_common.add_argument("--hardware", default="auto", choices=["auto", "cpu", "gpu"])
    worker_common.add_argument("--poll-interval", type=float, default=5.0)
    worker_common.add_argument("--validate-after-run", action="store_true")

    worker_start = worker_sub.add_parser("start", help="Start a persistent worker loop", parents=[worker_common])
    worker_start.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    worker_sub.add_parser("once", help="Claim and execute a single queued run", parents=[worker_common])

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    repository = LabRepository(settings)

    if args.command == "init":
        repository.init()
        print(f"Initialized lab at {settings.workspace_root}")
        return

    repository.init()

    if args.command == "task" and args.task_command == "new":
        task = repository.create_task(
            direction_slug=args.direction,
            title=args.title,
            kind=args.kind,
            description=args.description,
            owner=args.owner,
            priority=args.priority,
        )
        print(task.model_dump_json(indent=2))
        return

    if args.command == "run" and args.run_command == "start":
        try:
            validate_execution_request(
                requested_hardware=args.hardware,
                requested_kernel=args.kernel,
            )
        except ValueError as exc:
            parser.error(str(exc))
        run = repository.create_run(
            direction_slug=args.direction,
            name=args.name,
            range_start=args.start,
            range_end=args.end,
            kernel=args.kernel,
            owner=args.owner,
            hardware=args.hardware,
        )
        if not args.enqueue_only:
            run = execute_run(repository, run.id, checkpoint_interval=args.checkpoint_interval)
        print(run.model_dump_json(indent=2))
        return

    if args.command == "run" and args.run_command == "resume":
        run = execute_run(repository, args.run_id, checkpoint_interval=args.checkpoint_interval)
        print(run.model_dump_json(indent=2))
        return

    if args.command == "validate":
        run = validate_run(repository, args.run_id)
        print(run.model_dump_json(indent=2))
        return

    if args.command == "claim" and args.claim_command == "new":
        dependencies = [item.strip() for item in args.dependencies.split(",") if item.strip()]
        claim = repository.create_claim(
            direction_slug=args.direction,
            title=args.title,
            statement=args.statement,
            owner=args.owner,
            dependencies=dependencies,
            notes=args.notes,
        )
        print(claim.model_dump_json(indent=2))
        return

    if args.command == "claim" and args.claim_command == "link-run":
        link = repository.link_claim_run(
            claim_id=args.claim_id,
            run_id=args.run_id,
            relation=args.relation,
        )
        print(link.model_dump_json(indent=2))
        return

    if args.command == "direction" and args.direction_command == "review":
        review = repository.review_direction(args.slug)
        print(review.model_dump_json(indent=2))
        return

    if args.command == "report" and args.report_command == "generate":
        path = generate_report(repository)
        payload = {"path": str(Path(path).resolve())}
        print(json.dumps(payload, indent=2))
        return

    if args.command == "worker" and args.worker_command == "capabilities":
        print(json.dumps([item.model_dump() for item in discover_hardware()], indent=2))
        return

    if args.command == "worker" and args.worker_command in {"start", "once"}:
        # Set up file-based logging for worker processes
        log_dir = settings.workspace_root / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"worker-{args.name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        silence_numba_cuda_info()
        logging.getLogger("collatz_lab").info(
            "Worker %s starting (hardware=%s, poll=%.1fs)",
            args.name, args.hardware, args.poll_interval,
        )
        result = start_worker_loop(
            repository,
            name=args.name,
            role=args.role,
            hardware=args.hardware,
            poll_interval=args.poll_interval,
            validate_after_run=args.validate_after_run,
            once=args.worker_command == "once",
        )
        print(
            json.dumps(
                {
                    "worker": result.worker.model_dump(),
                    "processed_run_id": result.processed_run_id,
                },
                indent=2,
            )
        )
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
