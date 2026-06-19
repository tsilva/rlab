from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def database_url(use_direct: bool) -> str:
    load_dotenv()
    if use_direct:
        value = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    else:
        value = os.environ.get("DATABASE_URL") or os.environ.get("DIRECT_DATABASE_URL")
    if not value:
        raise SystemExit("DATABASE_URL or DIRECT_DATABASE_URL must be set in .env")
    return value


def connect(url: str):
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_one(cur, query: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    cur.execute(query, params or {})
    row = cur.fetchone()
    return dict(row) if row else {}


def fetch_all(cur, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cur.execute(query, params or {})
    return [dict(row) for row in cur.fetchall()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report Neon eval queue results.")
    parser.add_argument("--eval-profile", default="mario_level1_no_life_loss_v1")
    parser.add_argument("--stage", default="quick")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10007)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = {
        "eval_profile": args.eval_profile,
        "stage": args.stage,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "limit": args.limit,
    }
    conn = connect(database_url(args.direct))
    try:
        with conn.cursor() as cur:
            candidates = fetch_one(cur, "SELECT COUNT(*) AS count FROM checkpoint_candidates")
            statuses = fetch_all(
                cur,
                """
                SELECT status, COUNT(*) AS count
                FROM eval_jobs
                WHERE eval_profile = %(eval_profile)s
                  AND stage = %(stage)s
                  AND episodes = %(episodes)s
                  AND seed_start = %(seed_start)s
                GROUP BY status
                ORDER BY status
                """,
                params,
            )
            summary = fetch_one(
                cur,
                """
                SELECT
                  COUNT(*) AS result_count,
                  AVG(completion_rate) AS mean_completion_rate,
                  MAX(completion_rate) AS max_completion_rate,
                  AVG(max_x_max) AS mean_max_x_max,
                  MAX(max_x_max) AS best_max_x_max,
                  AVG(reward_mean) AS mean_reward_mean
                FROM eval_results
                WHERE eval_profile = %(eval_profile)s
                  AND stage = %(stage)s
                  AND episodes = %(episodes)s
                  AND seed_start = %(seed_start)s
                """,
                params,
            )
            runtime = fetch_one(
                cur,
                """
                SELECT
                  COUNT(*) AS timed_jobs,
                  AVG(EXTRACT(EPOCH FROM finished_at - started_at)) AS mean_seconds,
                  STDDEV_SAMP(EXTRACT(EPOCH FROM finished_at - started_at)) AS stddev_seconds,
                  MIN(EXTRACT(EPOCH FROM finished_at - started_at)) AS min_seconds,
                  MAX(EXTRACT(EPOCH FROM finished_at - started_at)) AS max_seconds
                FROM eval_jobs
                WHERE eval_profile = %(eval_profile)s
                  AND stage = %(stage)s
                  AND episodes = %(episodes)s
                  AND seed_start = %(seed_start)s
                  AND status = 'done'
                  AND started_at IS NOT NULL
                  AND finished_at IS NOT NULL
                """,
                params,
            )
            top = fetch_all(
                cur,
                """
                SELECT
                  c.artifact_ref,
                  c.run_name,
                  c.checkpoint_step,
                  r.completion_count,
                  r.completion_rate,
                  r.max_x_max,
                  r.reward_mean,
                  r.created_at
                FROM eval_results r
                JOIN checkpoint_candidates c ON c.id = r.candidate_id
                WHERE r.eval_profile = %(eval_profile)s
                  AND r.stage = %(stage)s
                  AND r.episodes = %(episodes)s
                  AND r.seed_start = %(seed_start)s
                ORDER BY r.completion_rate DESC, r.max_x_max DESC, r.reward_mean DESC
                LIMIT %(limit)s
                """,
                params,
            )

        print(f"# Eval Report: {args.eval_profile} / {args.stage}")
        print()
        print(f"- candidates: {int(candidates.get('count', 0))}")
        print(f"- episodes: {args.episodes}")
        print(f"- seed_start: {args.seed_start}")
        print("- job_status_counts:")
        if statuses:
            for row in statuses:
                print(f"  - {row['status']}: {int(row['count'])}")
        else:
            print("  - none")
        print(f"- result_count: {int(summary.get('result_count', 0) or 0)}")
        print(f"- mean_completion_rate: {float(summary.get('mean_completion_rate') or 0.0):.4f}")
        print(f"- max_completion_rate: {float(summary.get('max_completion_rate') or 0.0):.4f}")
        print(f"- mean_max_x_max: {float(summary.get('mean_max_x_max') or 0.0):.1f}")
        print(f"- best_max_x_max: {int(summary.get('best_max_x_max') or 0)}")
        print(f"- mean_reward_mean: {float(summary.get('mean_reward_mean') or 0.0):.3f}")
        print(
            "- runtime_seconds: "
            f"n={int(runtime.get('timed_jobs', 0) or 0)}, "
            f"mean={float(runtime.get('mean_seconds') or 0.0):.3f}, "
            f"std={float(runtime.get('stddev_seconds') or 0.0):.3f}, "
            f"min={float(runtime.get('min_seconds') or 0.0):.3f}, "
            f"max={float(runtime.get('max_seconds') or 0.0):.3f}"
        )
        print()
        print("## Top Results")
        if not top:
            print()
            print("No results for this eval selection.")
            return
        print()
        print("| rank | completion | max_x | reward_mean | step | run | artifact |")
        print("| ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for index, row in enumerate(top, start=1):
            run_name = str(row.get("run_name") or "")
            artifact_ref = str(row.get("artifact_ref") or "")
            print(
                f"| {index} "
                f"| {float(row['completion_rate']):.3f} ({int(row['completion_count'])}/{args.episodes}) "
                f"| {int(row['max_x_max'])} "
                f"| {float(row['reward_mean']):.2f} "
                f"| {int(row['checkpoint_step'] or 0)} "
                f"| `{run_name}` "
                f"| `{artifact_ref}` |"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
