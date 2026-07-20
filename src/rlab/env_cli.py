from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rlab.env_registry import (
    ENV_PROVIDERS,
    RLAB_PROVIDER,
    EnvProvider,
    resolve_env_id,
    resolve_env_provider,
)


def _editable_source_root(distribution: Any) -> Path | None:
    try:
        payload = json.loads(distribution.read_text("direct_url.json") or "")
    except AttributeError, json.JSONDecodeError, TypeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    dir_info = payload.get("dir_info")
    if not isinstance(dir_info, Mapping) or dir_info.get("editable") is not True:
        return None
    parsed = urlparse(str(payload.get("url") or ""))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_safe(item) for item in value]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(value)


def _provider_payload(
    provider: EnvProvider,
    *,
    provider_env_id: str | None = None,
) -> dict[str, Any]:
    contract = provider.constructor_contract
    payload: dict[str, Any] = {
        "provider_id": provider.provider_id,
        "import_name": provider.import_name,
        "distribution_name": provider.distribution_name,
        "env_ids": list(provider.env_ids),
        "allows_unregistered_env_ids": provider.allows_unregistered_env_ids,
        "supports_states": provider.supports_states,
        "external_rom_asset_strategy": provider.external_rom_asset_strategy,
        "requires_external_rom_asset": provider.requires_external_rom_asset,
        "constructor_contract": (
            {
                "kind": "fixed",
                "canonical_args": sorted(contract.canonical_args),
                "explicit_env_args": sorted(contract.explicit_env_args),
                "required_values": _json_safe(contract.required_values),
            }
            if contract is not None
            else {"kind": "dynamic"}
        ),
    }
    if provider_env_id is not None:
        payload["provider_env_id"] = provider_env_id
        payload["qualified_env_id"] = f"{provider.provider_id}:{provider_env_id}"
    return payload


def _space_payload(space: Any) -> dict[str, Any]:
    import numpy as np

    payload: dict[str, Any] = {
        "type": f"{type(space).__module__}.{type(space).__qualname__}",
    }
    shape = getattr(space, "shape", None)
    if shape is not None:
        payload["shape"] = [int(value) for value in shape]
    dtype = getattr(space, "dtype", None)
    if dtype is not None:
        payload["dtype"] = str(np.dtype(dtype))
    if hasattr(space, "n"):
        raw_n = np.asarray(space.n)
        payload["n"] = int(raw_n) if raw_n.ndim == 0 else raw_n.tolist()
    if hasattr(space, "nvec"):
        payload["nvec"] = np.asarray(space.nvec).tolist()
    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        payload["bounds"] = {
            "low_min": _json_safe(np.min(low).item()) if low.size else None,
            "low_max": _json_safe(np.max(low).item()) if low.size else None,
            "high_min": _json_safe(np.min(high).item()) if high.size else None,
            "high_max": _json_safe(np.max(high).item()) if high.size else None,
        }
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, Mapping):
        payload["spaces"] = {str(key): _space_payload(item) for key, item in spaces.items()}
    elif isinstance(spaces, Sequence):
        payload["spaces"] = [_space_payload(item) for item in spaces]
    return payload


def _copy_tree(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    return value


def _assert_unselected_equal(before: Any, after: Any, selected: Any, *, path: str = "obs") -> None:
    import numpy as np

    if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        if before.shape != after.shape:
            raise ValueError(f"{path} shape changed from {before.shape} to {after.shape}")
        if before.shape[:1] != selected.shape:
            raise ValueError(f"{path} does not contain one observation per lane")
        if not np.array_equal(before[~selected], after[~selected]):
            raise ValueError(f"{path} changed for an unselected masked-reset lane")
        return
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if before.keys() != after.keys():
            raise ValueError(f"{path} mapping keys changed during masked reset")
        for key in before:
            _assert_unselected_equal(before[key], after[key], selected, path=f"{path}.{key}")
        return
    if isinstance(before, tuple) and isinstance(after, tuple) and len(before) == len(after):
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            _assert_unselected_equal(left, right, selected, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} structure changed during masked reset")


def _diagnostic_start_ids(config: Any, n_envs: int, seed: int) -> list[str] | None:
    import numpy as np

    if config.states:
        if config.state_probs:
            probabilities = np.asarray(config.state_probs, dtype=np.float64)
            probabilities /= probabilities.sum()
            generator = np.random.default_rng(seed)
            return [
                str(value)
                for value in generator.choice(config.states, size=n_envs, p=probabilities)
            ]
        return [str(value) for value in config.states]
    if config.state:
        return [str(config.state) for _ in range(n_envs)]
    return None


def _reset_options(mask: Any, start_ids: Sequence[str] | None) -> dict[str, Any]:
    import numpy as np

    options: dict[str, Any] = {"reset_mask": np.asarray(mask, dtype=np.bool_).copy()}
    if start_ids is not None:
        options["start_ids"] = np.asarray(start_ids, dtype=object)
    return options


def _validate_observation_batch(space: Any, observations: Any, *, label: str) -> None:
    try:
        contained = bool(space.contains(observations))
    except Exception as exc:
        raise ValueError(f"{label} could not be validated against the observation space") from exc
    if not contained:
        raise ValueError(f"{label} is outside the declared batched observation space")


def _validate_columnar_infos(infos: Any, n_envs: int, *, label: str) -> Mapping[str, Any]:
    import numpy as np

    if not isinstance(infos, Mapping):
        raise TypeError(f"{label} must be a columnar mapping")

    def validate_column(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                validate_column(item, path=f"{path}.{key}")
            return
        array = np.asarray(value)
        if array.shape[:1] != (n_envs,):
            raise ValueError(f"{path} must contain one value per lane")

    for key, value in infos.items():
        validate_column(value, path=f"{label}.{key}")
    return infos


def _validate_native_step(
    result: Any,
    n_envs: int,
    *,
    observation_space: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import numpy as np

    if not isinstance(result, tuple) or len(result) != 5:
        raise TypeError("native provider step must return five values")
    observations, rewards, terminated, truncated, infos = result
    _validate_observation_batch(
        observation_space,
        observations,
        label="provider step observations",
    )
    for name, value in (("rewards", rewards), ("terminated", terminated), ("truncated", truncated)):
        array = np.asarray(value)
        if array.shape != (n_envs,):
            raise ValueError(f"provider {name} must have shape ({n_envs},), got {array.shape}")
        if name == "rewards" and not np.issubdtype(array.dtype, np.number):
            raise TypeError("provider rewards must have a numeric dtype")
        if name != "rewards" and array.dtype != np.dtype(np.bool_):
            raise TypeError(f"provider {name} must have boolean dtype")
    return observations, _validate_columnar_infos(infos, n_envs, label="provider step infos")


def _assert_reset_start_info(
    infos: Mapping[str, Any],
    selected: Any,
    *,
    n_envs: int,
    expected_starts: Sequence[str | None] | None,
) -> None:
    import numpy as np

    for key in ("start_id", "start_state", "state"):
        if key not in infos:
            continue
        values = np.asarray(infos[key], dtype=object)
        if values.shape != (n_envs,):
            raise ValueError(f"reset {key} must contain one value per lane")
        presence = infos.get(f"_{key}")
        if presence is not None:
            present = np.asarray(presence, dtype=np.bool_)
            if present.shape != (n_envs,) or np.any(selected & ~present):
                raise ValueError(f"reset {key} does not identify the selected lane")
        if any(values[index] is None for index in np.flatnonzero(selected)):
            raise ValueError(f"reset {key} omits the selected lane start")
        if expected_starts is not None:
            for index in np.flatnonzero(selected):
                expected = expected_starts[index]
                if expected is not None and str(values[index]) != str(expected):
                    raise ValueError(
                        f"reset {key} reported {values[index]!r} for lane {index}; "
                        f"expected {expected!r}"
                    )
        return
    raise ValueError("masked reset did not report start_id, start_state, or state")


def _record(
    report: dict[str, Any],
    name: str,
    status: str,
    detail: str,
    *,
    required: bool = True,
    evidence: str = "runtime",
) -> None:
    report["checks"].append(
        {
            "name": name,
            "required": required,
            "status": status,
            "evidence": evidence,
            "detail": detail,
        }
    )


def _finish_report(report: dict[str, Any]) -> bool:
    counts = {name: 0 for name in ("passed", "failed", "not_observable")}
    blocking: list[str] = []
    for check in report["checks"]:
        counts[check["status"]] += 1
        if check["status"] == "failed":
            blocking.append(check["name"])
        elif (
            check["required"]
            and check["status"] == "not_observable"
            and check["evidence"] != "pinned_provider_contract"
        ):
            blocking.append(check["name"])
    report["summary"] = {
        "ok": not blocking,
        "counts": counts,
        "blocking_checks": blocking,
    }
    return not blocking


def _check_report(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "declared": {},
        "resolved": {},
        "observed": {},
        "checks": [],
        "summary": {},
    }
    native_env: Any = None
    vec_env: Any = None
    stage = "recipe_validation"
    try:
        import importlib
        import importlib.metadata

        import gymnasium as gym
        import numpy as np

        from rlab.env import (
            assert_provider_runtime_available,
            bind_native_provider,
            make_native_provider,
            resolve_mixed_state_config,
        )
        from rlab.env_config import env_config_from_mapping
        from rlab.provider_config import provider_num_envs
        from rlab.recipe_documents import compose_train_document
        from rlab.training.sb3_vec_env import RlabVecEnv

        document = compose_train_document(
            args.goal_file,
            args.recipe_file,
            recipe_overrides=args.recipe_overrides,
        )
        train_config = document["train_config"]
        provider = resolve_env_provider(str(train_config["env_provider"]))
        config = env_config_from_mapping(train_config)
        n_envs = provider_num_envs(
            train_config,
            explicit_n_envs=train_config.get("n_envs"),
        )
        config = resolve_mixed_state_config(config, n_envs=n_envs)
        report["declared"] = _provider_payload(provider, provider_env_id=config.game)
        report["resolved"] = {
            "goal_file": str(args.goal_file),
            "recipe_file": str(args.recipe_file),
            "recipe_overrides": list(args.recipe_overrides),
            "qualified_env_id": f"{provider.provider_id}:{config.game}",
            "task_id": config.task.get("id"),
            "n_envs": n_envs,
            "seed": args.seed,
            "env_args": _json_safe(config.env_args),
        }
        _record(report, stage, "passed", "recipe and fixed constructor contract are valid")

        stage = "provider_import"
        distribution = importlib.metadata.distribution(provider.distribution_name)
        module = importlib.import_module(provider.import_name)
        distribution_root = Path(distribution.locate_file(".")).resolve()
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RuntimeError(f"provider module {provider.import_name!r} has no source path")
        module_path = Path(module_file).resolve()
        editable_source_root = (
            _editable_source_root(distribution)
            if provider.provider_id == RLAB_PROVIDER.provider_id
            else None
        )
        if not module_path.is_relative_to(distribution_root) and not (
            editable_source_root is not None and module_path.is_relative_to(editable_source_root)
        ):
            raise RuntimeError(
                f"provider module resolves outside its installed distribution: {module_path}"
            )
        report["observed"].update(
            {
                "distribution_version": distribution.version,
                "distribution_root": str(distribution_root),
                "module_path": str(module_path),
            }
        )
        if editable_source_root is not None:
            report["observed"]["editable_source_root"] = str(editable_source_root)
        _record(
            report, stage, "passed", f"loaded {provider.distribution_name} {distribution.version}"
        )

        stage = "runtime_availability"
        rom_binding = None
        if provider.requires_external_rom_asset:
            from rlab.rom_assets import rom_asset_manifest_for_game
            from rlab.rom_runtime import ensure_local_rom_binding

            rom_binding = ensure_local_rom_binding(
                rom_asset_manifest_for_game(config.game),
                game=config.game,
            )
        assert_provider_runtime_available(config, rom_binding=rom_binding)
        _record(report, stage, "passed", "provider runtime assets are available")

        stage = "native_construction"
        native_env, descriptor = make_native_provider(
            config,
            n_envs,
            rom_binding=rom_binding,
        )
        report["observed"].update(
            {
                "vector_type": f"{type(native_env).__module__}.{type(native_env).__qualname__}",
                "num_envs": int(native_env.num_envs),
                "native_observation_space": _space_payload(descriptor.native_observation_space),
                "native_action_space": _space_payload(descriptor.native_action_space),
                "signals": {
                    name: {
                        "dtype": str(spec.dtype),
                        "shape": list(spec.shape),
                        "available_on_reset": spec.available_on_reset,
                        "available_on_step": spec.available_on_step,
                    }
                    for name, spec in sorted(descriptor.signal_schema.items())
                },
                "start_catalog": list(descriptor.start_catalog),
                "lane_start_ids": list(descriptor.lane_start_ids),
                "render_support": list(descriptor.render_support),
                "autoreset_mode": descriptor.autoreset_mode,
                "observation_buffer_depth": descriptor.observation_buffer_depth,
            }
        )
        _record(report, stage, "passed", "constructed and described native vector provider")

        stage = "native_step"
        full_mask = np.ones(n_envs, dtype=np.bool_)
        start_ids = _diagnostic_start_ids(config, n_envs, args.seed)
        observations, reset_infos = native_env.reset(
            seed=[args.seed + lane for lane in range(n_envs)],
            options=_reset_options(full_mask, start_ids),
        )
        _validate_observation_batch(
            native_env.observation_space,
            observations,
            label="provider reset observations",
        )
        _validate_columnar_infos(reset_infos, n_envs, label="provider reset infos")
        native_env.action_space.seed(args.seed)
        stepped_observations, _step_infos = _validate_native_step(
            native_env.step(native_env.action_space.sample()),
            n_envs,
            observation_space=native_env.observation_space,
        )
        _record(
            report, stage, "passed", "native reset and seeded vector step satisfy the batch shape"
        )

        stage = "visible_masked_reset"
        contract_evidence = (
            "pinned_provider_contract" if provider.constructor_contract is not None else "runtime"
        )
        if n_envs < 2:
            _record(
                report,
                stage,
                "not_observable",
                "at least two lanes are required to observe preservation of an unselected lane",
                evidence=contract_evidence,
            )
        else:
            before_reset = _copy_tree(stepped_observations)
            selected = np.zeros(n_envs, dtype=np.bool_)
            selected[0] = True
            selected_starts = None
            if start_ids is not None:
                selected_starts = [start_ids[0], *([None] * (n_envs - 1))]
            reset_observations, masked_infos = native_env.reset(
                seed=[args.seed + n_envs, *([None] * (n_envs - 1))],
                options=_reset_options(selected, selected_starts),
            )
            _validate_observation_batch(
                native_env.observation_space,
                reset_observations,
                label="provider masked-reset observations",
            )
            _validate_columnar_infos(masked_infos, n_envs, label="provider masked-reset infos")
            _assert_unselected_equal(before_reset, reset_observations, selected)
            if descriptor.start_catalog:
                _assert_reset_start_info(
                    masked_infos,
                    selected,
                    n_envs=n_envs,
                    expected_starts=selected_starts,
                )
                _record(
                    report,
                    "masked_reset_start_info",
                    "passed",
                    "masked reset reports the selected lane start",
                )
            else:
                _record(
                    report,
                    "masked_reset_start_info",
                    "not_observable",
                    "provider has no configured start catalog",
                    required=False,
                )
            _record(
                report,
                stage,
                "passed",
                "visible observations for every unselected lane were preserved",
            )

        _record(
            report,
            "hidden_masked_reset_state",
            "not_observable",
            "emulator, RNG, frame-stack, counter, and sticky-action internals are not black-box observable",
            evidence=contract_evidence,
        )

        stage = "rlab_runtime_step"
        native_env.reset(
            seed=[args.seed + 2 * n_envs + lane for lane in range(n_envs)],
            options=_reset_options(full_mask, start_ids),
        )
        runtime = None
        try:
            runtime = bind_native_provider(
                config,
                n_envs=n_envs,
                seed=args.seed,
                native_env=native_env,
                descriptor=descriptor,
            )
            native_env = None
            vec_env = RlabVecEnv(runtime)
        except Exception:
            if runtime is None:
                native_env = None
            else:
                runtime.close()
            raise
        vec_env.seed(args.seed)
        vec_env.reset()
        policy_batch_space = gym.vector.utils.batch_space(vec_env.action_space, n_envs)
        policy_batch_space.seed(args.seed)
        vec_env.step(policy_batch_space.sample())
        _record(report, stage, "passed", "task binding and one rlab/SB3-facade step succeeded")
    except Exception as exc:
        _record(report, stage, "failed", f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_error: Exception | None = None
        try:
            if vec_env is not None:
                vec_env.close()
            elif native_env is not None:
                native_env.close()
        except Exception as exc:
            cleanup_error = exc
        if cleanup_error is None:
            _record(report, "cleanup", "passed", "environment resources were closed")
        else:
            _record(
                report,
                "cleanup",
                "failed",
                f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
    _finish_report(report)
    return report


def _print_human_report(report: Mapping[str, Any]) -> None:
    declared = report.get("declared") or {}
    resolved = report.get("resolved") or {}
    if declared:
        print(f"provider: {declared.get('provider_id')}")
    if resolved:
        print(f"environment: {resolved.get('qualified_env_id')}")
    for check in report["checks"]:
        print(
            f"{check['status'].upper():>14}  {check['name']}: {check['detail']} "
            f"[{check['evidence']}]"
        )
    summary = report["summary"]
    print(f"environment preflight: {'passed' if summary['ok'] else 'failed'}")


def _cmd_list(args: argparse.Namespace) -> int:
    providers = [_provider_payload(ENV_PROVIDERS[key]) for key in sorted(ENV_PROVIDERS)]
    if args.json:
        print(json.dumps({"schema_version": 1, "providers": providers}, indent=2, sort_keys=True))
        return 0
    for provider in providers:
        environments = ", ".join(provider["env_ids"]) or "dynamic registered IDs"
        contract = provider["constructor_contract"]["kind"]
        print(f"{provider['provider_id']}: {environments} (constructor={contract})")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    resolved = resolve_env_id(args.env_id)
    provider = resolve_env_provider(resolved.provider_id)
    payload = {
        "schema_version": 1,
        "environment": _provider_payload(provider, provider_env_id=resolved.provider_env_id),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload["environment"], indent=2, sort_keys=True))
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    if args.json:
        with redirect_stdout(sys.stderr):
            report = _check_report(args)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        report = _check_report(args)
        _print_human_report(report)
    return 0 if report["summary"]["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlab env", description="Inspect environment providers.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List declared providers and environments.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=_cmd_list)

    inspect_parser = commands.add_parser("inspect", help="Inspect one declared environment.")
    inspect_parser.add_argument("env_id", metavar="<provider:environment>")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(handler=_cmd_inspect)

    preflight_parser = commands.add_parser(
        "preflight", help="Run a recipe-backed environment preflight."
    )
    preflight_parser.add_argument("--goal-file", type=Path, required=True)
    preflight_parser.add_argument(
        "--recipe-file",
        type=Path,
        required=True,
        help="Launchable recipe under the selected goal's recipes directory.",
    )
    preflight_parser.add_argument("--set", dest="recipe_overrides", action="append", default=[])
    preflight_parser.add_argument("--seed", type=int, default=0)
    preflight_parser.add_argument("--json", action="store_true")
    preflight_parser.set_defaults(handler=_cmd_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
