# 06 - Run Name Path Traversal

Severity: Medium
Confidence: Medium
Subagent: Boyle

## Problem

Queue job `run_name` flows from `train_runner.normalize_train_config()` into `--run-name`, then `train.py` computes `run_dir = default_run_dir(args.run_name, args.runs_dir)`. Today `default_run_dir()` is a raw `os.path.join`, so names like `../escape` or `/tmp/escape` can write checkpoints, best models, run descriptions, and W&B aux dirs outside the intended `runs_dir`.

Affected files:

- `src/rlab/env.py:752-753`
- `src/rlab/train.py:57-62`
- `src/rlab/artifacts.py:298-309`
- `src/rlab/train_runner.py:37-43`

## Desired state

`run_name` is treated as an experiment identifier, not a path. Training run directories always resolve beneath `runs_dir`, regardless of CLI or queue input. Invalid names fail early with a clear `ValueError`, or are converted through one shared slug policy with collision handling. Prefer strict rejection for path separators, absolute paths, `.`, `..`, and empty normalized names.

## Implementation plan

1. Add shared helpers near `default_run_dir()` in `src/rlab/env.py`:
   - `sanitize_run_name(value: str) -> str`.
   - `safe_run_dir(run_name: str, runs_dir: str | Path = "runs") -> Path`.
   - Keep allowed chars aligned with existing artifact name sanitization: `[A-Za-z0-9_.-]`.
   - Explicitly reject path separators, absolute paths, `.`, `..`, and values whose sanitized form differs if using strict validation.
2. Change `default_run_dir()` to delegate to the safe helper and return `str` only if existing callers expect strings.
3. Internally resolve `Path(runs_dir)` and verify the final resolved path is equal to or inside resolved `runs_dir`.
4. Update `train.py` to use the safe helper result for all run-local writes.
5. Update `train_runner.normalize_train_config()` to validate `run_name` before building the subprocess command.
6. Update `train_runner.collect_result_metadata()` to use the same safe helper instead of `Path(runs_dir) / run_name`.
7. Do not change W&B display names unless required. Preserve the original validated `run_name` in metadata/config where useful.

## Tests/verification

Add focused unit tests, likely in `tests/test_core_helpers.py` and `tests/test_job_queue_runner.py`:

- `default_run_dir("candidate", tmp_runs)` returns a path under `tmp_runs`.
- Reject traversal: `../escape`, `a/../b`, `a/b`, `..`, `.`, and empty string.
- Reject or normalize absolute paths such as `/tmp/escape`.
- `normalize_train_config({"id": 1, "run_name": "../escape"})` raises before command construction.
- `collect_result_metadata()` does not read marker files outside `runs_dir` when given a malicious run name.
- Keep existing artifact URI tests passing; S3 artifact paths already sanitize run names separately.

Run:

```bash
uv run pytest tests/test_core_helpers.py tests/test_job_queue_runner.py
```

## Rollout notes

This may reject historical ad hoc run names containing slashes. That is desirable for training output directories, but check launch manifests and queued train jobs before rollout. If legitimate existing run names contain unsupported characters, migrate them to safe slugs in manifests rather than weakening the directory invariant.

Do not auto-move already-created escaped directories in this patch. Treat cleanup as a separate operator task.

## Open questions

- Should invalid run names be rejected strictly, or slugified automatically?
- Should the queue store both `display_run_name` and `run_name_slug` long term?
- Are there pending queue rows with slash-containing `run_name` values that need migration before deploying validation?
