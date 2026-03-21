# Worker Queue Contract

## Purpose

Document the execution model for local Collatz runs so the lab can move from direct execution to queued, hardware-aware dispatch without changing the core run metadata shape.

## Principles

- CPU remains the default execution path.
- GPU is optional and must be selected explicitly or by policy.
- Every run stays reproducible from saved metadata.
- Queue state must not replace run state; it only schedules work.

## Proposed States

- `queued`: the run is waiting for a worker.
- `leased`: a worker has claimed the run but not finished it.
- `running`: the worker is actively processing the interval.
- `completed`: the run finished successfully.
- `validated`: the result matched the independent validator.
- `failed`: the run could not complete or validation diverged.

## Hardware Selection

- Prefer CPU for short or validation-oriented runs.
- Prefer GPU only for large sweeps or kernels that can actually benefit from throughput.
- Keep the selected hardware visible in the run record.
- Avoid silent promotion between hardware classes; any change should be recorded.

## Worker Responsibilities

- fetch the next eligible run from the queue;
- lock or lease the run before execution;
- write checkpoints often enough to survive interruption;
- persist final metrics and artifacts before completion;
- hand completed runs to validation as a separate step.

## Failure Handling

- interrupted work should resume from the last checkpoint;
- a lease timeout should return the run to the queue;
- validation must be able to reject a run independently of the worker that produced it;
- hardware-specific failures should be recorded in the run summary and artifacts.

## Acceptance

- a CPU run and a GPU run should share the same record format;
- the queue should be observable from the dashboard and CLI;
- a resumed run should not lose processed values or checkpoints;
- the worker choice should be explainable from the recorded metadata.
