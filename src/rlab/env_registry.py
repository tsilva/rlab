from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvProvider:
    provider_id: str
    import_name: str
    env_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedEnvId:
    qualified_id: str
    provider_id: str
    provider_env_id: str
    import_name: str


STABLE_RETRO_TURBO_PROVIDER = EnvProvider(
    provider_id="stable-retro-turbo",
    import_name="stable_retro",
    env_ids=("SuperMarioBros-Nes-v0",),
)

ENV_PROVIDERS: dict[str, EnvProvider] = {
    STABLE_RETRO_TURBO_PROVIDER.provider_id: STABLE_RETRO_TURBO_PROVIDER,
}

LEGACY_PROVIDER_ALIASES = {
    "stable_retro": STABLE_RETRO_TURBO_PROVIDER.provider_id,
}


def registered_env_ids() -> tuple[str, ...]:
    return tuple(
        f"{provider.provider_id}:{env_id}"
        for provider in ENV_PROVIDERS.values()
        for env_id in provider.env_ids
    )


def qualify_env_id(provider_id: str, provider_env_id: str) -> str:
    provider_id = str(provider_id).strip()
    provider_env_id = str(provider_env_id).strip()
    if not provider_id:
        raise ValueError("environment provider id is required")
    if not provider_env_id:
        raise ValueError("provider environment id is required")
    provider_id = LEGACY_PROVIDER_ALIASES.get(provider_id, provider_id)
    if provider_id not in ENV_PROVIDERS:
        known = ", ".join(sorted(ENV_PROVIDERS))
        raise ValueError(f"unknown environment provider {provider_id!r}; known providers: {known}")
    return f"{provider_id}:{provider_env_id}"


def resolve_env_id(env_id: str) -> ResolvedEnvId:
    value = str(env_id).strip()
    if ":" not in value:
        raise ValueError(
            "environment env_id must be fully qualified as <provider>:<env>, "
            f"got {value!r}"
        )
    provider_id, provider_env_id = value.split(":", 1)
    provider_id = LEGACY_PROVIDER_ALIASES.get(provider_id.strip(), provider_id.strip())
    provider_env_id = provider_env_id.strip()
    if not provider_id or not provider_env_id:
        raise ValueError(
            "environment env_id must be fully qualified as <provider>:<env>, "
            f"got {value!r}"
        )
    provider = ENV_PROVIDERS.get(provider_id)
    if provider is None:
        known = ", ".join(sorted(ENV_PROVIDERS))
        raise ValueError(f"unknown environment provider {provider_id!r}; known providers: {known}")
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
