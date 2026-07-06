"""
cache_service.py — Application-level caching wrapper.
Provides centralized caching for embeddings, features, market data, and predictions.
"""

import logging
from typing import Any, Callable

from django.core.cache import cache

logger = logging.getLogger(__name__)


class CacheService:
    @staticmethod
    def get_or_set(key: str, fallback_func: Callable[[], Any], timeout: int = 300) -> Any:
        """
        Get value from cache, or compute using fallback_func and save to cache.
        timeout is in seconds (default 5 minutes).
        """
        value = cache.get(key)
        if value is not None:
            logger.debug("Cache hit for key: %s", key)
            return value

        logger.debug("Cache miss for key: %s", key)
        try:
            value = fallback_func()
            cache.set(key, value, timeout=timeout)
            return value
        except Exception as exc:
            logger.warning("Error computing value for cache key %s: %s", key, exc)
            raise

    @staticmethod
    def delete(key: str) -> None:
        """Invalidate a specific cache key."""
        cache.delete(key)
        logger.debug("Invalidated cache key: %s", key)

    @staticmethod
    def invalidate_prefix(prefix: str) -> None:
        """
        Invalidate all keys starting with prefix.
        Note: Depending on the cache backend, this might require specific implementation.
        For LocMemCache, clearing entirely or managing specific keys is often easier.
        """
        # In a real Redis setup, we might use keys(), but LocMem doesn't support wildcards easily.
        # We will clear the whole cache for simplicity if prefix is needed,
        # or rely on exact key invalidation.
        cache.clear()
        logger.info("Cleared cache (fallback for prefix invalidation: %s)", prefix)
