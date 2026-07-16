#!/usr/bin/env python3
"""Stable skill entrypoint for the repo-owned training launch workflow."""

from rlab.training_launch import main, safe_error


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        from rlab.training_launch import emit

        emit("workflow_error", error=safe_error(exc))
        exit_code = 1
    raise SystemExit(exit_code)
