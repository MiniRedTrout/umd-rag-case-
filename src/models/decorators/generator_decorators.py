from functools import wraps
from typing import Any, AsyncGenerator, Generator, TypeVar

from loguru import logger
from openai import AuthenticationError

from src.models.errors import is_content_filtered
from src.utils.embedding_token_limit import is_request_too_large

T = TypeVar("T")


def generator_retry_on_error(max_retries: int = 3):
    def decorator(gen_func):
        @wraps(gen_func)
        def wrapper(self, *args, **kwargs) -> Generator[T, Any, None]:
            attempt = 1

            while True:
                try:
                    for item in gen_func(self, *args, **kwargs):
                        yield item
                    break
                except AuthenticationError as e:
                    if attempt <= max_retries:
                        logger.warning(
                            f"Received error (auth) {str(e)} in {gen_func.__name__}. "
                            f"Attempt {attempt}/{max_retries}. Updating API key..."
                        )
                        self._update_api_key()
                        attempt += 1
                        continue
                    else:
                        logger.error(f"Maximum number of attempts ({max_retries}) exceeded")
                        raise e
                except Exception as e:
                    if is_content_filtered(e):
                        logger.warning(f"Content blocked in {gen_func.__name__}: {e}")
                        raise
                    if (
                        is_request_too_large(e)
                        or (attempt > max_retries)
                    ):
                        logger.error(e)
                        raise e
                    logger.warning(
                            f"Received error (exception) {str(e)} in {gen_func.__name__}. "
                        f"Attempt {attempt}/{max_retries}"
                    )
                    attempt += 1
        return wrapper

    return decorator


def async_generator_retry_on_error(max_retries: int = 3):
    def decorator(gen_func):
        @wraps(gen_func)
        async def wrapper(self, *args, **kwargs) -> AsyncGenerator[T, Any]:
            attempt = 1

            while True:
                try:
                    async for item in gen_func(self, *args, **kwargs):
                        yield item
                    break

                except AuthenticationError as e:
                    if attempt <= max_retries:
                        logger.warning(
                            f"Received error (auth) {str(e)} in {gen_func.__name__}. "
                            f"Attempt {attempt}/{max_retries}. Updating API key..."
                        )
                        self._update_api_key()
                        attempt += 1
                    else:
                        logger.error(
                            f"Maximum number of attempts ({max_retries}) exceeded")
                        raise e
                except Exception as e:
                    if is_content_filtered(e):
                        logger.warning(f"Content blocked in {gen_func.__name__}: {e}")
                        raise
                    if (
                        is_request_too_large(e)
                        or (attempt > max_retries)
                    ):
                        logger.error(e)
                        raise e
                    logger.warning(
                            f"Received error (exception) {str(e)} in {gen_func.__name__}. "
                        f"Attempt {attempt}/{max_retries}"
                    )
                    attempt += 1
        return wrapper

    return decorator
