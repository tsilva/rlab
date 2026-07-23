from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from rlab.job_queue import json_arg
from rlab.telemetry_integrity import (
    EVIDENCE_VERSION,
    build_eval_scope_exact,
    build_run_final_exact,
    build_training_success_scope_exact,
    require_comparable_run_facts,
    sha256_json,
)
from rlab.metric_names import METRICS_SCHEMA_VERSION


def exact_evaluation_contract(
    *,
    train_config: Mapping[str, object],
    execution_contract: Mapping[str, object],
) -> dict[str, object]:
    """Expand the execution descriptor into the contract-key comparison surface."""

    environment = dict(execution_contract.get("environment") or {})
    manifest = dict(execution_contract.get("manifest") or {})
    runtime = str(
        execution_contract.get("runtime_image_ref")
        or train_config.get("runtime_image_ref")
        or ""
    )
    if "sha256:" not in runtime:
        raise ValueError("exact evaluation requires an immutable runtime image digest")
    return {
        "canonical_goal_sha256": train_config.get("goal_sha256"),
        "effective_goal_contract_sha256": train_config.get(
            "effective_goal_contract_sha256"
        ),
        "goal_slug": train_config.get("goal_slug"),
        "recipe_sha256": execution_contract.get("recipe_sha256")
        or train_config.get("recipe_sha256"),
        "policy_bundle_sha256": train_config.get("policy_bundle_sha256"),
        "runtime_image_digest": runtime,
        "dependency_lock_sha256": train_config.get("dependency_lock_sha256")
        or train_config.get("uv_lock_sha256"),
        "evaluator_implementation_sha256": train_config.get(
            "evaluator_implementation_sha256"
        )
        or train_config.get("source_sha"),
        "metrics_schema_version": int(
            train_config.get("metrics_schema_version") or METRICS_SCHEMA_VERSION
        ),
        "seed_protocol": execution_contract.get("seed_protocol"),
        "n_envs": execution_contract.get("n_envs"),
        "episodes": execution_contract.get("episodes"),
        "max_steps": execution_contract.get("max_steps"),
        "deterministic": False,
        "action_sampling": "stochastic",
        "environment": environment,
        "observation": {
            key: train_config[key]
            for key in (
                "observation_type",
                "frame_stack",
                "screen_width",
                "screen_height",
                "environment_contract_sha256",
            )
            if key in train_config
        }
        or {"environment": environment},
        "action": {
            "sampling": "stochastic",
            "action_space": train_config.get("action_space"),
            "training_backend": train_config.get("training_backend"),
        },
        "preprocessing": {
            key: train_config[key]
            for key in (
                "frame_skip",
                "frame_stack",
                "normalize_observations",
                "screen_width",
                "screen_height",
            )
            if key in train_config
        }
        or {"identity": "provider-default"},
        "reward": {
            "program_name": train_config.get("reward_program_name")
            or train_config.get("reward_shape"),
            "program_revision": train_config.get("reward_program_revision"),
            "program_sha256": train_config.get("reward_program_sha256")
            or train_config.get("reward_shape_sha256"),
        },
        "events": {
            key: value
            for key, value in train_config.items()
            if str(key).startswith(("event_", "success_", "failure_"))
        }
        or {"identity": train_config.get("event_contract_sha256")},
        "starts": manifest or {"environment": environment.get("state")},
        "termination": {
            "max_steps": execution_contract.get("max_steps"),
            "semantics": train_config.get("termination_contract_sha256")
            or "provider-contract",
        },
        "assets": dict(
            execution_contract.get("asset")
            or train_config.get("rom_asset_manifest")
            or {"identity": "no-external-asset"}
        ),
    }


def insert_evidence_scope(
    cur,
    *,
    train_job_id: int,
    evidence: Mapping[str, Any],
    root_sha256: str | None = None,
) -> str:
    scope_kind = str(evidence.get("scope_kind") or "")
    scope_sha256 = str(evidence.get("scope_sha256") or sha256_json(evidence))
    scope_key = str(
        evidence.get("execution_key")
        or evidence.get("success_event", {}).get("event_id")
        or scope_sha256
    )
    cur.execute(
        """
        INSERT INTO telemetry_evidence_scopes (
          train_job_id, scope_kind, scope_key, scope_sha256,
          root_sha256, evidence_json, state, finalized_at
        ) VALUES (
          %(run)s, %(kind)s, %(key)s, %(sha256)s, %(root)s,
          %(evidence)s, 'exact', now()
        )
        ON CONFLICT (train_job_id, scope_kind, scope_key) DO UPDATE
        SET evidence_json = CASE
          WHEN telemetry_evidence_scopes.scope_sha256 = EXCLUDED.scope_sha256
          THEN EXCLUDED.evidence_json
          ELSE telemetry_evidence_scopes.evidence_json
        END
        RETURNING scope_sha256
        """,
        {
            "run": int(train_job_id),
            "kind": scope_kind,
            "key": scope_key,
            "sha256": scope_sha256,
            "root": root_sha256,
            "evidence": json_arg(dict(evidence)),
        },
    )
    observed = str(cur.fetchone()["scope_sha256"])
    if observed != scope_sha256:
        raise RuntimeError("authoritative evidence scope conflicts with prior evidence")
    return scope_sha256


def persist_evidence_scope(
    conn,
    *,
    train_job_id: int,
    evidence: Mapping[str, Any],
    root_sha256: str | None = None,
) -> str:
    scope_kind = str(evidence.get("scope_kind") or "")
    if scope_kind not in {"eval_scope_exact", "training_success_scope_exact"}:
        raise ValueError("unsupported authoritative evidence scope")
    scope_sha256 = str(evidence.get("scope_sha256") or sha256_json(evidence))
    scope_key = str(
        evidence.get("execution_key")
        or evidence.get("success_event", {}).get("event_id")
        or scope_sha256
    )
    with conn:
        with conn.cursor() as cur:
            return insert_evidence_scope(
                cur,
                train_job_id=int(train_job_id),
                evidence=evidence,
                root_sha256=root_sha256,
            )


def materialize_eval_scope(
    conn,
    *,
    train_job_id: int,
    checkpoint: Mapping[str, object],
    evaluation_contract: Mapping[str, object],
    episode_manifest: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    acceptance_rule: Mapping[str, object],
    execution_key: str,
    attestation: Mapping[str, object],
) -> str:
    evidence = build_eval_scope_exact(
        checkpoint=checkpoint,
        evaluation_contract=evaluation_contract,
        episode_manifest=episode_manifest,
        results=results,
        acceptance_rule=acceptance_rule,
        execution_key=execution_key,
        attestation=attestation,
    )
    return persist_evidence_scope(
        conn,
        train_job_id=train_job_id,
        evidence=evidence,
        root_sha256=str(checkpoint["sha256"]),
    )


def materialize_training_success_scope(
    conn,
    *,
    train_job_id: int,
    contract: Mapping[str, object],
    success_event: Mapping[str, object],
    policy_artifact: Mapping[str, object],
) -> str:
    evidence = build_training_success_scope_exact(
        contract=contract,
        success_event=success_event,
        policy_artifact=policy_artifact,
    )
    return persist_evidence_scope(
        conn,
        train_job_id=train_job_id,
        evidence=evidence,
        root_sha256=str(policy_artifact["sha256"]),
    )


def persist_run_final(
    conn,
    *,
    train_job_id: int,
    dimensions: Mapping[str, object],
    metrics: Mapping[str, object],
    seed: int,
    cohort_manifest: Mapping[str, object],
) -> dict[str, object]:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.root_sha256, i.state_json, i.exact, i.classification
                FROM telemetry_archive_roots r
                JOIN telemetry_integrity i
                  ON i.train_job_id = r.train_job_id
                 AND i.telemetry_generation = r.telemetry_generation
                WHERE r.train_job_id = %(run)s AND r.root_kind = 'exact'
                FOR UPDATE
                """,
                {"run": int(train_job_id)},
            )
            proof = cur.fetchone()
            if not proof or not bool(proof["exact"]):
                raise RuntimeError("run_final_exact requires exact frozen telemetry integrity")
            integrity = {
                **dict(proof["state_json"]),
                "exact": bool(proof["exact"]),
                "classification": str(proof["classification"]),
            }
            facts = build_run_final_exact(
                archive_root_sha256=str(proof["root_sha256"]),
                dimensions=dimensions,
                metrics=metrics,
                seed=int(seed),
                cohort_manifest=cohort_manifest,
                integrity=integrity,
            )
            cur.execute(
                """
                INSERT INTO telemetry_run_facts (
                  train_job_id, scope_sha256, archive_root_sha256,
                  comparability_sha256, cohort_manifest_sha256, seed,
                  rank_metric, rank_direction, facts_json
                ) VALUES (
                  %(run)s, %(scope)s, %(root)s, %(comparison)s, %(cohort)s,
                  %(seed)s, %(metric)s, %(direction)s, %(facts)s
                )
                ON CONFLICT (train_job_id) DO UPDATE
                SET facts_json = CASE
                  WHEN telemetry_run_facts.scope_sha256 = EXCLUDED.scope_sha256
                  THEN EXCLUDED.facts_json
                  ELSE telemetry_run_facts.facts_json
                END
                RETURNING scope_sha256
                """,
                {
                    "run": int(train_job_id),
                    "scope": facts["scope_sha256"],
                    "root": facts["archive_root_sha256"],
                    "comparison": facts["comparability_sha256"],
                    "cohort": facts["cohort_manifest_sha256"],
                    "seed": int(seed),
                    "metric": str(facts["dimensions"]["rank_metric"]),
                    "direction": str(facts["dimensions"]["rank_direction"]),
                    "facts": json_arg(facts),
                },
            )
            if str(cur.fetchone()["scope_sha256"]) != str(facts["scope_sha256"]):
                raise RuntimeError("run_final_exact conflicts with existing authoritative facts")
    from rlab.telemetry_reducer import reduce_run_integrity

    reduce_run_integrity(conn, train_job_id=int(train_job_id))
    return facts


def authoritative_run_facts(
    conn,
    *,
    goal_slug: str | None = None,
    reward_shape: str | None = None,
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.facts_json
            FROM telemetry_run_facts f
            JOIN telemetry_integrity i ON i.train_job_id = f.train_job_id
            WHERE i.exact = TRUE
              AND i.classification = 'intact_with_proof'
              AND (%(goal)s IS NULL OR f.facts_json->'dimensions'->>'goal_slug' = %(goal)s)
              AND (
                %(reward_shape)s IS NULL
                OR f.facts_json->'dimensions'->>'reward_program_name' = %(reward_shape)s
              )
            ORDER BY f.comparability_sha256, f.cohort_manifest_sha256, f.seed
            """,
            {"goal": goal_slug, "reward_shape": reward_shape},
        )
        facts = [dict(row["facts_json"]) for row in cur.fetchall()]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        grouped[
            (
                str(fact["comparability_sha256"]),
                str(fact["cohort_manifest_sha256"]),
            )
        ].append(fact)
    for cohort in grouped.values():
        require_comparable_run_facts(cohort, require_complete_cohort=True)
    return facts


def authoritative_checkpoint_evidence(
    conn,
    *,
    goal_slug: str | None = None,
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.evidence_json
            FROM telemetry_evidence_scopes e
            JOIN telemetry_integrity i ON i.train_job_id = e.train_job_id
            WHERE e.scope_kind = 'eval_scope_exact'
              AND e.state = 'exact'
              AND i.exact = TRUE
              AND i.classification = 'intact_with_proof'
              AND (
                %(goal)s IS NULL
                OR e.evidence_json->'evaluation_contract'->>'goal_slug' = %(goal)s
              )
            ORDER BY e.scope_sha256
            """,
            {"goal": goal_slug},
        )
        return [dict(row["evidence_json"]) for row in cur.fetchall()]


def training_facts_by_wandb_run_id(conn, *, wandb_run_id: str) -> dict[str, Any]:
    """Compatibility lookup for orchestrators; the returned authority is not W&B."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.facts_json
            FROM telemetry_run_facts f
            JOIN telemetry_integrity i ON i.train_job_id = f.train_job_id
            WHERE f.facts_json->>'wandb_run_id' = %(wandb_run_id)s
              AND i.exact = TRUE
              AND i.classification = 'intact_with_proof'
            """,
            {"wandb_run_id": str(wandb_run_id)},
        )
        rows = cur.fetchall()
    if len(rows) != 1:
        raise RuntimeError("authoritative run_final_exact evidence is missing or ambiguous")
    return dict(rows[0]["facts_json"])
