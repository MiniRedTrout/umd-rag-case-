from functools import wraps
from typing import Any, Callable, TypeVar

from loguru import logger
from openai import AuthenticationError

from src.models.errors import is_content_filtered
from src.utils.embedding_token_limit import is_request_too_large

T = TypeVar("T")


def retry_on_error(max_retries: int = 3):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> T:
            attempt = 1
            while True:
                try:
                    return func(self, *args, **kwargs)
                except AuthenticationError as e:
                    if attempt > max_retries:
                        logger.error(f"Maximum number of attempts ({max_retries}) exceeded. Error: {e}")
                        raise e
                    if attempt <= max_retries:
                        logger.warning(
                            f"Received error {str(e)} in {func.__name__}. "
                            f"Attempt {attempt}/{max_retries}. Updating API key..."
                        )
                        self._update_api_key()
                        attempt += 1
                except Exception as e:
                    if is_content_filtered(e):
                        logger.warning(f"Content blocked in {func.__name__}: {e}")
                        raise
                    if (
                        is_request_too_large(e)
                        or (attempt > max_retries)
                    ):
                        logger.error(e)
                        raise e
                    attempt += 1
                    logger.warning(
                        f"Received error {str(e)} in {func.__name__}. "
                        f"Attempt {attempt}/{max_retries}."
                    )
        return wrapper

    return decorator


def async_retry_on_error(max_retries: int = 3):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any) -> T:
            attempt = 1
            while True:
                try:
                    return await func(self, *args, **kwargs)
                except AuthenticationError as e:
                    if attempt > max_retries:
                        logger.error(
                                f"Maximum number of attempts ({max_retries}) exceeded. Error: {e}")
                        raise e
                    if attempt <= max_retries:
                        logger.warning(
                            f"Received error {str(e)} in {func.__name__}. "
                            f"Attempt {attempt}/{max_retries}. Updating API key..."
                        )
                        self._update_api_key()
                        attempt += 1
                except Exception as e:
                    if is_content_filtered(e):
                        logger.warning(f"Content blocked in {func.__name__}: {e}")
                        raise
                    if (
                        is_request_too_large(e)
                        or (attempt > max_retries)
                    ):
                        logger.error(e)
                        raise e
                    attempt += 1
                    logger.warning(
                        f"Received error {str(e)} in {func.__name__}. "
                        f"Attempt {attempt}/{max_retries}."
                    )
        return wrapper

    return decorator
