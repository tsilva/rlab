from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


EXTERNAL_ROM_ASSET_NONE = "none"
STABLE_RETRO_DIRECT_PATH_V1 = "stable_retro_direct_path_v1"
EXTERNAL_ROM_ASSET_STRATEGIES = frozenset(
    {EXTERNAL_ROM_ASSET_NONE, STABLE_RETRO_DIRECT_PATH_V1}
)


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
    external_rom_asset_strategy: str = EXTERNAL_ROM_ASSET_NONE
    allows_unregistered_env_ids: bool = False
    constructor_contract: ProviderConstructorContract | None = None

    def __post_init__(self) -> None:
        if self.external_rom_asset_strategy not in EXTERNAL_ROM_ASSET_STRATEGIES:
            raise ValueError(
                f"unsupported external ROM asset strategy: {self.external_rom_asset_strategy}"
            )

    @property
    def requires_external_rom_asset(self) -> bool:
        return self.external_rom_asset_strategy != EXTERNAL_ROM_ASSET_NONE


@dataclass(frozen=True)
class ResolvedEnvId:
    qualified_id: str
    provider_id: str
    provider_env_id: str
    import_name: str


@dataclass(frozen=True)
class CanonicalEnvironmentIdentity:
    game_family: str
    wandb_project: str
    env_id_game_family_fallback: bool = True
    env_id_wandb_project_fallback: bool = True


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
    external_rom_asset_strategy=STABLE_RETRO_DIRECT_PATH_V1,
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
                "state_catalog",
                "sticky_action_prob",
            }
        ),
        explicit_env_args=frozenset(
            {
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
        required_values={},
    ),
)

SUPERMARIOBROS_NES_TURBO_PROVIDER = EnvProvider(
    provider_id="supermariobrosnes-turbo",
    import_name="supermariobrosnes_turbo",
    distribution_name="supermariobrosnes-turbo",
    env_ids=("SuperMarioBros-Nes-v0",),
    external_rom_asset_strategy=STABLE_RETRO_DIRECT_PATH_V1,
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
                "state_catalog",
                "sticky_action_prob",
            }
        ),
        explicit_env_args=frozenset(
            {
                "action_set",
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
                "state_dir",
                "use_restricted_actions",
            }
        ),
        required_values={"action_set": None},
    ),
)

BREAKOUT_TURBO_ENV_PROVIDER = EnvProvider(
    provider_id="breakout-turbo-env",
    import_name="breakout_turbo_env",
    distribution_name="breakout-turbo-env",
    env_ids=("Breakout-Atari2600-v0", "BreakoutTurbo-v0"),
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
                "state_catalog",
                "sticky_action_prob",
            }
        ),
        explicit_env_args=frozenset(
            {
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
        required_values={},
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

RLAB_PROVIDER = EnvProvider(
    provider_id="rlab",
    import_name="rlab",
    distribution_name="rlab",
    env_ids=("Bandit-v0",),
    supports_states=False,
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset({"game", "num_envs"}),
        explicit_env_args=frozenset({"autoreset_mode"}),
        required_values={"autoreset_mode": "disabled"},
    ),
)

ENV_PROVIDERS: dict[str, EnvProvider] = {
    RLAB_PROVIDER.provider_id: RLAB_PROVIDER,
    BREAKOUT_TURBO_ENV_PROVIDER.provider_id: BREAKOUT_TURBO_ENV_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER.provider_id: STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id: SUPERMARIOBROS_NES_TURBO_PROVIDER,
    ALE_PY_PROVIDER.provider_id: ALE_PY_PROVIDER,
    GYMNASIUM_PROVIDER.provider_id: GYMNASIUM_PROVIDER,
}

# Provider-neutral public identity used by W&B metadata, project routing, and
# published model identities. Publication must use this explicit registry;
# provider-local environment ids are not parsed into public identity fields.
# The env-id fallback flags preserve historical reads where old metadata omitted
# or carried a stale provider. ALE's short ids historically inferred a family
# but not a W&B project when the provider mapping did not match.
CANONICAL_ENVIRONMENT_IDENTITIES: Mapping[
    tuple[str, str], CanonicalEnvironmentIdentity
] = MappingProxyType(
    {
        ("rlab", "Bandit-v0"): CanonicalEnvironmentIdentity("Bandit", "Bandit-v0"),
        (
            "supermariobrosnes-turbo",
            "SuperMarioBros-Nes-v0",
        ): CanonicalEnvironmentIdentity("NES-SuperMarioBros", "SuperMarioBros-Nes-v0"),
        (
            "stable-retro-turbo",
            "SuperMarioBros-Nes-v0",
        ): CanonicalEnvironmentIdentity("NES-SuperMarioBros", "SuperMarioBros-Nes-v0"),
        (
            "stable-retro-turbo",
            "SuperMarioBros3-Nes-v0",
        ): CanonicalEnvironmentIdentity("NES-SuperMarioBros3", "SuperMarioBros3-Nes-v0"),
        (
            "breakout-turbo-env",
            "BreakoutTurbo-v0",
        ): CanonicalEnvironmentIdentity("Atari2600-Breakout", "Breakout-Atari2600-v0"),
        (
            "breakout-turbo-env",
            "Breakout-Atari2600-v0",
        ): CanonicalEnvironmentIdentity("Atari2600-Breakout", "Breakout-Atari2600-v0"),
        ("ale-py", "breakout"): CanonicalEnvironmentIdentity(
            "Atari2600-Breakout",
            "Breakout-Atari2600-v0",
            env_id_wandb_project_fallback=False,
        ),
        (
            "stable-retro-turbo",
            "Breakout-Atari2600-v0",
        ): CanonicalEnvironmentIdentity("Atari2600-Breakout", "Breakout-Atari2600-v0"),
        ("ale-py", "ms_pacman"): CanonicalEnvironmentIdentity(
            "Atari2600-MsPacman",
            "MsPacman-Atari2600-v0",
            env_id_wandb_project_fallback=False,
        ),
        (
            "stable-retro-turbo",
            "MsPacman-Atari2600-v0",
        ): CanonicalEnvironmentIdentity("Atari2600-MsPacman", "MsPacman-Atari2600-v0"),
    }
)

STABLE_RETRO_ATARI_ENV_IDS = frozenset(
    {"Breakout-Atari2600-v0", "MsPacman-Atari2600-v0"}
)


def _environment_identity(provider_id: object, env_id: object) -> tuple[str, str]:
    provider = str(provider_id or "").strip()
    environment = str(env_id or "").strip()
    if not provider and ":" in environment:
        provider, environment = environment.split(":", 1)
    return provider, environment


def _canonical_identity_by_env_id(
    env_id: str,
    *,
    fallback_field: str,
) -> CanonicalEnvironmentIdentity | None:
    matches = {
        identity
        for (_provider, registered_env_id), identity in CANONICAL_ENVIRONMENT_IDENTITIES.items()
        if registered_env_id == env_id and getattr(identity, fallback_field)
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def _canonical_environment_identity(
    provider_id: object,
    env_id: object,
    *,
    fallback_field: str,
    allow_env_id_fallback: bool = True,
) -> tuple[str, CanonicalEnvironmentIdentity | None]:
    """Resolve a registered public identity while preserving historical reads."""

    provider, environment = _environment_identity(provider_id, env_id)
    identity = CANONICAL_ENVIRONMENT_IDENTITIES.get((provider, environment))
    if identity is None and allow_env_id_fallback:
        identity = _canonical_identity_by_env_id(
            environment,
            fallback_field=fallback_field,
        )
    return environment, identity


def _fallback_game_family(env_id: str, *, fallback: str) -> str:
    value = env_id or fallback
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    return re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-") or "environment"


def game_family_for_environment(
    provider_id: object,
    env_id: object,
    *,
    strict: bool = False,
    fallback: str = "environment",
) -> str:
    """Return the provider-neutral public family for an environment.

    Training metadata retains the historical fallback for arbitrary Gymnasium
    environments. Publication passes ``strict=True`` and therefore requires an
    explicit registered family rather than guessing a public model identity.
    """

    environment, identity = _canonical_environment_identity(
        provider_id,
        env_id,
        fallback_field="env_id_game_family_fallback",
        allow_env_id_fallback=not strict,
    )
    if identity is not None:
        return identity.game_family
    if strict:
        provider, _environment = _environment_identity(provider_id, env_id)
        qualified = f"{provider}:{environment}" if provider else environment
        raise ValueError(
            f"environment {qualified!r} has no registered canonical game family"
        )
    return _fallback_game_family(environment, fallback=fallback)


def wandb_project_for_environment(
    provider_id: object,
    env_id: object,
    *,
    fallback: str,
) -> str:
    """Return the registered W&B project with historical providerless fallback."""

    environment, identity = _canonical_environment_identity(
        provider_id,
        env_id,
        fallback_field="env_id_wandb_project_fallback",
    )
    if identity is not None:
        return identity.wandb_project
    return environment or fallback


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
