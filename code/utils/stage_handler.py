from __future__ import annotations

import functools
from typing import Any, Callable

from exceptions import PipelineError


def stage_handler(
    stage_name: str,
    maps: dict[type[Exception], type[PipelineError]],
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except PipelineError:
                raise
            except Exception as exc:
                for source_type, target_type in maps.items():
                    if isinstance(exc, source_type):
                        raise target_type(stage=stage_name, cause=exc) from exc
                raise
        return wrapper
    return decorator