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
    parser.add_argument("--profile", default="mario-level1-quick")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = {"profile": args.profile, "limit": args.limit}
    conn = connect(database_url(args.direct))
    try:
        with conn.cursor() as cur:
            statuses = fetch_all(
                cur,
                """
                SELECT status, COUNT(*) AS count
                FROM eval_jobs
                WHERE profile_id = %(profile)s
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
                  AVG((metrics_json->>'completion_rate')::DOUBLE PRECISION) AS mean_completion_rate,
                  MAX((metrics_json->>'completion_rate')::DOUBLE PRECISION) AS max_completion_rate,
                  AVG((metrics_json->>'max_x_max')::DOUBLE PRECISION) AS mean_max_x_max,
                  MAX((metrics_json->>'max_x_max')::DOUBLE PRECISION) AS best_max_x_max,
                  AVG((metrics_json->>'reward_mean')::DOUBLE PRECISION) AS mean_reward_mean
                FROM eval_results
                WHERE profile_id = %(profile)s
                  AND status = 'succeeded'
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
                WHERE profile_id = %(profile)s
                  AND status = 'succeeded'
                  AND started_at IS NOT NULL
                  AND finished_at IS NOT NULL
                """,
                params,
            )
            top = fetch_all(
                cur,
                """
                SELECT
                  r.candidate_label,
                  r.model_ref,
                  r.output_path,
                  (r.metrics_json->>'episodes')::INTEGER AS episodes,
                  (r.metrics_json->>'completion_count')::INTEGER AS completion_count,
                  (r.metrics_json->>'completion_rate')::DOUBLE PRECISION AS completion_rate,
                  (r.metrics_json->>'max_x_max')::INTEGER AS max_x_max,
                  (r.metrics_json->>'reward_mean')::DOUBLE PRECISION AS reward_mean,
                  r.created_at
                FROM eval_results r
                WHERE r.profile_id = %(profile)s
                  AND r.status = 'succeeded'
                ORDER BY completion_rate DESC, reward_mean DESC, max_x_max DESC
                LIMIT %(limit)s
                """,
                params,
            )

        print(f"# Eval Report: {args.profile}")
        print()
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
        print("| rank | completion | max_x | reward_mean | candidate | model | output |")
        print("| ---: | ---: | ---: | ---: | --- | --- | --- |")
        for index, row in enumerate(top, start=1):
            episodes = int(row.get("episodes") or 0)
            completion_count = int(row.get("completion_count") or 0)
            print(
                f"| {index} "
                f"| {float(row['completion_rate']):.3f} ({completion_count}/{episodes}) "
                f"| {int(row['max_x_max'])} "
                f"| {float(row['reward_mean']):.2f} "
                f"| `{row.get('candidate_label') or ''}` "
                f"| `{row.get('model_ref') or ''}` "
                f"| `{row.get('output_path') or ''}` |"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
