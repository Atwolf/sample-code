"""Redis-backed ADK memory service construction."""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass
from typing import Any

from ag_ui_adk import SessionManager

from app.config import Settings


@dataclass(frozen=True)
class MemoryServices:
    session_manager: SessionManager
    memory_service: Any


def _httpx_verify_value(value: bool | str) -> bool | str:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


def _memory_api_client(*, client_config: Any, verify: bool | str) -> Any:
    import httpx
    from agent_memory_client import MemoryAPIClient, __version__

    class VerifiedMemoryAPIClient(MemoryAPIClient):
        def __init__(self, config: Any) -> None:
            self.config = config
            self._client = httpx.AsyncClient(
                base_url=config.base_url,
                timeout=config.timeout,
                verify=verify,
                headers={
                    "User-Agent": f"agent-memory-client/{__version__}",
                    "X-Client-Version": __version__,
                },
            )

    return VerifiedMemoryAPIClient(client_config)


def build_memory_services(settings: Settings) -> MemoryServices:
    """Construct the Redis ADK working-memory and long-term-memory services."""

    from adk_redis.memory import (
        RedisLongTermMemoryService,
        RedisLongTermMemoryServiceConfig,
    )
    from adk_redis.sessions import (
        RedisWorkingMemorySessionService,
        RedisWorkingMemorySessionServiceConfig,
    )
    from agent_memory_client import MemoryClientConfig

    verify = _httpx_verify_value(settings.redis_memory_tls_verify)

    class VerifiedRedisWorkingMemorySessionService(RedisWorkingMemorySessionService):
        def _get_client(self) -> Any:
            client_config = MemoryClientConfig(
                base_url=self._config.api_base_url,
                timeout=self._config.timeout,
                default_namespace=self._config.default_namespace,
                default_model_name=self._config.model_name,
                default_context_window_max=self._config.context_window_max,
            )
            return _memory_api_client(client_config=client_config, verify=verify)

    class VerifiedRedisLongTermMemoryService(RedisLongTermMemoryService):
        @cached_property
        def _client(self) -> Any:
            client_config = MemoryClientConfig(
                base_url=self._config.api_base_url,
                timeout=self._config.timeout,
                default_namespace=self._config.default_namespace,
                default_model_name=self._config.model_name,
                default_context_window_max=self._config.context_window_max,
            )
            return _memory_api_client(client_config=client_config, verify=verify)

    session_service = VerifiedRedisWorkingMemorySessionService(
        config=RedisWorkingMemorySessionServiceConfig(
            api_base_url=settings.redis_memory_api_base_url,
            default_namespace=settings.app_name,
            model_name=settings.default_model,
            context_window_max=settings.redis_memory_context_window_max,
        )
    )
    memory_service = VerifiedRedisLongTermMemoryService(
        config=RedisLongTermMemoryServiceConfig(
            api_base_url=settings.redis_memory_api_base_url,
            default_namespace=settings.app_name,
            extraction_strategy=settings.redis_memory_extraction_strategy,
            recency_boost=settings.redis_memory_recency_boost,
            semantic_weight=settings.redis_memory_semantic_weight,
            recency_weight=settings.redis_memory_recency_weight,
        )
    )
    session_manager = SessionManager(
        session_service=session_service,
        memory_service=memory_service,
        session_timeout_seconds=settings.session_timeout_seconds,
        cleanup_interval_seconds=settings.session_cleanup_interval_seconds,
        max_sessions_per_user=None,
        # Working memory is retained in Redis. Long-term memory is updated by
        # the ADK after_agent_callback after each run, not by TTL cleanup.
        delete_session_on_cleanup=False,
        save_session_to_memory_on_cleanup=False,
        use_thread_id_as_session_id=True,
    )
    return MemoryServices(
        session_manager=session_manager,
        memory_service=memory_service,
    )
