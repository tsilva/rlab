from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ProviderConstructorContract:
    canonical_args: frozenset[str]
    explicit_env_args: frozenset[str]
    required_values: Mapping[str, object]

    def __post_init__(self) -> None:
        overlap = self.canonical_args & self.explicit_env_args
        if overlap:
            raise ValueError(f"provider constructor argument ownership overlaps: {sorted(overlap)}")
        unknown_required = set(self.required_values) - set(self.explicit_env_args)
        if unknown_required:
            raise ValueError(
                "provider required values are not explicit env args: "
                f"{sorted(unknown_required)}"
            )
        object.__setattr__(self, "required_values", MappingProxyType(dict(self.required_values)))


@dataclass(frozen=True)
class EnvProvider:
    provider_id: str
    import_name: str
    distribution_name: str
    env_ids: tuple[str, ...]
    supports_states: bool = True
    uses_stable_retro_roms: bool = False
    allows_unregistered_env_ids: bool = False
    constructor_contract: ProviderConstructorContract | None = None


@dataclass(frozen=True)
class ResolvedEnvId:
    qualified_id: str
    provider_id: str
    provider_env_id: str
    import_name: str


STABLE_RETRO_TURBO_PROVIDER = EnvProvider(
    provider_id="stable-retro-turbo",
    import_name="stable_retro",
    distribution_name="stable-retro-turbo",
    env_ids=(
        "SuperMarioBros-Nes-v0",
        "SuperMarioBros3-Nes-v0",
        "Breakout-Atari2600-v0",
        "MsPacman-Atari2600-v0",
    ),
    uses_stable_retro_roms=True,
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset(
            {
                "frame_skip",
                "game",
                "maxpool_last_two",
                "num_envs",
                "obs_crop",
                "obs_crop_fill",
                "obs_crop_mode",
                "obs_resize",
                "obs_resize_algorithm",
                "state",
                "sticky_action_prob",
            }
        ),
        explicit_env_args=frozenset(
            {
                "autoreset_mode",
                "done_on",
                "frame_stack",
                "info",
                "info_filter",
                "inttype",
                "noop_reset_max",
                "num_threads",
                "obs_copy",
                "obs_grayscale",
                "obs_layout",
                "obs_type",
                "players",
                "record",
                "render_mode",
                "reward_clip",
                "rom_path",
                "scenario",
                "use_fire_reset",
                "use_restricted_actions",
            }
        ),
        required_values={"autoreset_mode": "disabled", "done_on": None},
    ),
)

SUPERMARIOBROS_NES_TURBO_PROVIDER = EnvProvider(
    provider_id="supermariobrosnes-turbo",
    import_name="supermariobrosnes_turbo",
    distribution_name="supermariobrosnes-turbo",
    env_ids=("SuperMarioBros-Nes-v0",),
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset(
            {
                "frame_skip",
                "game",
                "maxpool_last_two",
                "num_envs",
                "obs_crop",
                "obs_crop_fill",
                "obs_crop_mode",
                "obs_resize",
                "obs_resize_algorithm",
                "state",
                "sticky_action_prob",
            }
        ),
        explicit_env_args=frozenset(
            {
                "autoreset_mode",
                "done_on",
                "frame_stack",
                "info",
                "info_filter",
                "inttype",
                "noop_reset_max",
                "num_threads",
                "obs_copy",
                "obs_grayscale",
                "obs_layout",
                "obs_type",
                "players",
                "record",
                "render_mode",
                "reward_clip",
                "rom_path",
                "scenario",
                "use_restricted_actions",
            }
        ),
        required_values={"autoreset_mode": "disabled", "done_on": None},
    ),
)

ALE_PY_PROVIDER = EnvProvider(
    provider_id="ale-py",
    import_name="ale_py",
    distribution_name="ale-py",
    env_ids=("breakout", "ms_pacman"),
    supports_states=False,
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset(
            {
                "frameskip",
                "game",
                "img_height",
                "img_width",
                "maxpool",
                "num_envs",
                "repeat_action_probability",
            }
        ),
        explicit_env_args=frozenset(
            {
                "autoreset_mode",
                "batch_size",
                "continuous",
                "continuous_action_threshold",
                "episodic_life",
                "full_action_space",
                "grayscale",
                "life_loss_info",
                "max_num_frames_per_episode",
                "noop_max",
                "num_threads",
                "reward_clipping",
                "stack_num",
                "thread_affinity_offset",
                "use_fire_reset",
            }
        ),
        required_values={"autoreset_mode": "next_step"},
    ),
)

GYMNASIUM_PROVIDER = EnvProvider(
    provider_id="gymnasium",
    import_name="gymnasium",
    distribution_name="gymnasium",
    env_ids=(),
    supports_states=False,
    allows_unregistered_env_ids=True,
)

ENV_PROVIDERS: dict[str, EnvProvider] = {
    STABLE_RETRO_TURBO_PROVIDER.provider_id: STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id: SUPERMARIOBROS_NES_TURBO_PROVIDER,
    ALE_PY_PROVIDER.provider_id: ALE_PY_PROVIDER,
    GYMNASIUM_PROVIDER.provider_id: GYMNASIUM_PROVIDER,
}

STABLE_RETRO_ATARI_ENV_IDS = frozenset(
    {"Breakout-Atari2600-v0", "MsPacman-Atari2600-v0"}
)


def is_stable_retro_atari_env(provider_id: str, game: str) -> bool:
    return (
        str(provider_id) == STABLE_RETRO_TURBO_PROVIDER.provider_id
        and str(game) in STABLE_RETRO_ATARI_ENV_IDS
    )


def env_supports_states(provider_id: str, game: str) -> bool:
    provider = resolve_env_provider(provider_id)
    return provider.supports_states


def registered_env_ids() -> tuple[str, ...]:
    return tuple(
        f"{provider.provider_id}:{env_id}"
        for provider in ENV_PROVIDERS.values()
        for env_id in provider.env_ids
    )


def resolve_env_provider(provider_id: str) -> EnvProvider:
    provider_id = str(provider_id).strip()
    if not provider_id:
        raise ValueError("environment provider id is required")
    provider = ENV_PROVIDERS.get(provider_id)
    if provider is None:
        known = ", ".join(sorted(ENV_PROVIDERS))
        raise ValueError(f"unknown environment provider {provider_id!r}; known providers: {known}")
    return provider


def validate_provider_constructor_args(
    provider_id: str,
    env_args: Any,
    *,
    label: str,
) -> None:
    provider = resolve_env_provider(provider_id)
    if not isinstance(env_args, Mapping):
        raise ValueError(f"{label} must explicitly define provider arguments")
    contract = provider.constructor_contract
    if contract is None:
        return
    actual_args = set(env_args)
    missing_args = sorted(contract.explicit_env_args - actual_args)
    if missing_args:
        raise ValueError(
            f"{label} missing explicit {provider.provider_id} constructor argument(s): "
            + ", ".join(missing_args)
        )
    unexpected_args = sorted(actual_args - contract.explicit_env_args)
    if unexpected_args:
        raise ValueError(
            f"{label} has unexpected or canonically-owned {provider.provider_id} "
            f"constructor argument(s): {', '.join(unexpected_args)}"
        )
    for key, expected in contract.required_values.items():
        actual = env_args.get(key)
        if actual == expected:
            continue
        expected_text = "null" if expected is None else repr(expected)
        raise ValueError(f"{label}.{key} must be {expected_text}; got {actual!r}")


def qualify_env_id(provider_id: str, provider_env_id: str) -> str:
    provider_env_id = str(provider_env_id).strip()
    if not provider_env_id:
        raise ValueError("provider environment id is required")
    provider = resolve_env_provider(provider_id)
    if not provider.allows_unregistered_env_ids and provider_env_id not in provider.env_ids:
        known = ", ".join(provider.env_ids)
        raise ValueError(
            f"provider {provider.provider_id!r} does not register environment {provider_env_id!r}; "
            f"known envs: {known}"
        )
    return f"{provider.provider_id}:{provider_env_id}"


def resolve_env_id(env_id: str) -> ResolvedEnvId:
    value = str(env_id).strip()
    if ":" not in value:
        raise ValueError(
            f"environment env_id must be fully qualified as <provider>:<env>, got {value!r}"
        )
    provider_id, provider_env_id = value.split(":", 1)
    provider_id = provider_id.strip()
    provider_env_id = provider_env_id.strip()
    if not provider_id or not provider_env_id:
        raise ValueError(
            f"environment env_id must be fully qualified as <provider>:<env>, got {value!r}"
        )
    provider = resolve_env_provider(provider_id)
    if not provider.allows_unregistered_env_ids and provider_env_id not in provider.env_ids:
        known = ", ".join(provider.env_ids)
        raise ValueError(
            f"provider {provider_id!r} does not register environment {provider_env_id!r}; "
            f"known envs: {known}"
        )
    return ResolvedEnvId(
        qualified_id=f"{provider.provider_id}:{provider_env_id}",
        provider_id=provider.provider_id,
        provider_env_id=provider_env_id,
        import_name=provider.import_name,
    )
