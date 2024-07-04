import time
from typing import Any

from ahcore.utils.io import get_logger

logger = get_logger("time_it")

TIME_IT_ENABLE = True


def time_it(func: Any) -> Any:
    """Decorator function to log execution speed of functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not TIME_IT_ENABLE:
            return func(*args, **kwargs)

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # Determining the function name for logging
        if hasattr(func, "__qualname__"):
            func_name = func.__qualname__
        else:
            func_name = func.__name__

        # Log the function execution time
        logger.info(f"{func_name} took {elapsed_time:.2f} seconds to execute.")

        # Optionally log the arguments and keyword arguments
        if args or kwargs:
            logger.debug(f"{func_name} called with args: {args}, kwargs: {kwargs}")

        return result

    return wrapper
