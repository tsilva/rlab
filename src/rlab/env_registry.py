from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvProvider:
    provider_id: str
    import_name: str
    env_ids: tuple[str, ...]
    supports_states: bool = True
    uses_stable_retro_roms: bool = False


@dataclass(frozen=True)
class ResolvedEnvId:
    qualified_id: str
    provider_id: str
    provider_env_id: str
    import_name: str


STABLE_RETRO_TURBO_PROVIDER = EnvProvider(
    provider_id="stable-retro-turbo",
    import_name="stable_retro",
    env_ids=("SuperMarioBros-Nes-v0", "SuperMarioBros3-Nes-v0"),
    uses_stable_retro_roms=True,
)

SUPERMARIOBROS_NES_TURBO_PROVIDER = EnvProvider(
    provider_id="supermariobrosnes-turbo",
    import_name="supermariobrosnes_turbo",
    env_ids=("SuperMarioBros-Nes-v0",),
)

ALE_PY_PROVIDER = EnvProvider(
    provider_id="ale-py",
    import_name="ale_py",
    env_ids=("breakout",),
    supports_states=False,
)

ENV_PROVIDERS: dict[str, EnvProvider] = {
    STABLE_RETRO_TURBO_PROVIDER.provider_id: STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id: SUPERMARIOBROS_NES_TURBO_PROVIDER,
    ALE_PY_PROVIDER.provider_id: ALE_PY_PROVIDER,
}


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


def qualify_env_id(provider_id: str, provider_env_id: str) -> str:
    provider_env_id = str(provider_env_id).strip()
    if not provider_env_id:
        raise ValueError("provider environment id is required")
    provider = resolve_env_provider(provider_id)
    if provider_env_id not in provider.env_ids:
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
            "environment env_id must be fully qualified as <provider>:<env>, "
            f"got {value!r}"
        )
    provider_id, provider_env_id = value.split(":", 1)
    provider_id = provider_id.strip()
    provider_env_id = provider_env_id.strip()
    if not provider_id or not provider_env_id:
        raise ValueError(
            "environment env_id must be fully qualified as <provider>:<env>, "
            f"got {value!r}"
        )
    provider = resolve_env_provider(provider_id)
    if provider_env_id not in provider.env_ids:
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
