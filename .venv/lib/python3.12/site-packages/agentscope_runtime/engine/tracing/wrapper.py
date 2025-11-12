# -*- coding: utf-8 -*-
# pylint:disable=too-many-statements,typevar-name-incorrect-variance
import inspect
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    Optional,
    Union,
)
from typing import AsyncIterable, AsyncIterator, Tuple, TypeVar

from pydantic import BaseModel
from openai.types.chat import ChatCompletionChunk

from .base import Tracer, TraceType
from .local_logging_handler import LocalLogHandler

T = TypeVar("T", covariant=True)


async def aenumerate(
    asequence: AsyncIterable[T],
    start: int = 0,
) -> AsyncIterator[Tuple[int, T]]:
    """Asynchronously enumerate an async iterator from a given start value.

    Args:
        asequence (AsyncIterable[T]): The async iterable to enumerate.
        start (int): The starting value for enumeration. Defaults to 0.

    Yields:
        Tuple[int, T]: A tuple containing the index and the item from the
        async iterable.
    """
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1


def trace(
    trace_type: Union[TraceType, str],
    tracer: Optional[Tracer] = Tracer(
        handlers=[LocalLogHandler(enable_console=True)],
    ),
) -> Any:
    """Decorator for tracing function execution.

    Args:
        trace_type (Union[TraceType, str]): The type of trace event.
        tracer (Optional[Tracer]): The tracer instance to use. Defaults to
            a new Tracer with LocalLogHandler.

    Returns:
        Any: The decorated function with tracing capabilities.
    """
    if isinstance(trace_type, str):
        trace_type = TraceType(trace_type)

    def task_wrapper(func: Any) -> Any:
        """Wrapper function that applies tracing to the target function.

        Args:
            func (Any): The function to be traced.

        Returns:
            Any: The wrapped function with appropriate tracing logic.
        """

        async def async_exec(*args: Any, **kwargs: Any) -> Any:
            """Execute async function with tracing.

            Args:
                *args (Any): Positional arguments for the function.
                **kwargs (Any): Keyword arguments for the function.

            Returns:
                Any: The result of the traced function.
            """
            start_payload = _get_start_payload(args, kwargs)
            with tracer.event(
                trace_type,
                payload=start_payload,
                **kwargs,
            ) as event:
                kwargs = kwargs if kwargs is not None else {}
                kwargs["trace_event"] = event
                result = await func(*args, **kwargs)
                event.on_end(payload=_obj_to_dict(result))
                return result

        def sync_exec(*args: Any, **kwargs: Any) -> Any:
            """Execute sync function with tracing.

            Args:
                *args (Any): Positional arguments for the function.
                **kwargs (Any): Keyword arguments for the function.

            Returns:
                Any: The result of the traced function.
            """
            start_payload = _get_start_payload(args, kwargs)
            with tracer.event(
                trace_type,
                payload=start_payload,
                **kwargs,
            ) as event:
                kwargs = kwargs if kwargs is not None else {}
                kwargs["trace_event"] = event
                result = func(*args, **kwargs)
                event.on_end(payload=_obj_to_dict(result))
                return result

        @wraps(func)
        async def async_iter_task(
            *args: Any,
            **kwargs: Any,
        ) -> AsyncGenerator[T, None]:
            """Execute async generator function with tracing.

            Args:
                *args (Any): Positional arguments for the function.
                **kwargs (Any): Keyword arguments for the function.

            Yields:
                T: Items from the traced async generator.
            """
            start_payload = _get_start_payload(args, kwargs)
            with tracer.event(
                trace_type,
                payload=start_payload,
                **kwargs,
            ) as event:
                kwargs = kwargs if kwargs is not None else {}
                kwargs["trace_event"] = event
                cumulated = []

                async def iter_entry() -> AsyncGenerator[T, None]:
                    """Internal async generator for processing items.

                    Yields:
                        T: Items from the original generator with tracing.
                    """
                    try:
                        async for i, resp in aenumerate(
                            func(*args, **kwargs),
                        ):  # type: ignore
                            if i == 0:
                                event.on_log(
                                    "",
                                    **{
                                        "step_suffix": "first_resp",
                                        "payload": resp.model_dump(),
                                    },
                                )
                            # todo: support more components
                            if isinstance(resp, ChatCompletionChunk):
                                if len(resp.choices) > 0:
                                    cumulated.append(resp)
                                    if (
                                        resp.choices[0].finish_reason
                                        is not None
                                    ):
                                        if (
                                            resp.choices[0].finish_reason
                                            == "stop"
                                        ):
                                            step_suffix = "last_resp"
                                        else:
                                            step_suffix = resp.choices[
                                                0
                                            ].finish_reason
                                        event.on_log(
                                            "",
                                            **{
                                                "step_suffix": step_suffix,
                                                "payload": resp.model_dump(),
                                            },
                                        )
                                elif resp.usage:
                                    cumulated.append(resp)

                            yield resp
                    except Exception as e:
                        raise e

                try:
                    async for resp in iter_entry():
                        yield resp

                except Exception as e:
                    raise e

        @wraps(func)
        def iter_task(*args: Any, **kwargs: Any) -> Iterable[T]:
            """Execute generator function with tracing.

            Args:
                *args (Any): Positional arguments for the function.
                **kwargs (Any): Keyword arguments for the function.

            Yields:
                T: Items from the traced generator.
            """
            start_payload = _get_start_payload(args, kwargs)
            with tracer.event(
                trace_type,
                payload=start_payload,
                **kwargs,
            ) as event:
                cumulated = []
                try:
                    kwargs = kwargs if kwargs is not None else {}
                    kwargs["trace_event"] = event
                    for i, resp in enumerate(func(*args, **kwargs)):
                        if i == 0:
                            event.on_log(
                                "",
                                **{
                                    "step_suffix": "first_resp",
                                    "payload": resp.model_dump(),
                                },
                            )
                        # todo: support more components
                        if len(resp.choices) > 0:
                            cumulated.append(resp)
                            if resp.choices[0].finish_reason is not None:
                                if resp.choices[0].finish_reason == "stop":
                                    step_suffix = "last_resp"
                                else:
                                    step_suffix = resp.choices[0].finish_reason
                                event.on_log(
                                    "",
                                    **{
                                        "step_suffix": step_suffix,
                                        "payload": resp.model_dump(),
                                    },
                                )
                        elif resp.usage:
                            cumulated.append(resp)

                        yield resp
                except Exception as e:
                    raise e

        if inspect.isasyncgenfunction(func):
            return async_iter_task
        elif inspect.isgeneratorfunction(func):
            return iter_task
        elif inspect.iscoroutinefunction(func):
            return async_exec
        else:
            return sync_exec

    return task_wrapper


def _get_start_payload(args: Any, kwargs: Any) -> Dict:
    """Extract and format the start payload from function arguments.

    Args:
        args (Any): Positional arguments from the function call.
        kwargs (Any): Keyword arguments from the function call.

    Returns:
        Dict: The formatted start payload for tracing.
    """
    dict_args = {}
    if isinstance(args, tuple) and len(args) > 1:
        dict_args = _obj_to_dict(args[1:])

    dict_kwargs = _obj_to_dict(kwargs)
    dict_kwargs = {
        key: value
        for key, value in dict_kwargs.items()
        if not key.startswith("trace_")
    }

    merged = {}
    if dict_args:
        if isinstance(dict_args, list):
            for item in dict_args:
                if isinstance(item, dict):
                    merged.update(item)
        elif isinstance(dict_args, dict):
            merged.update(dict_args)

    if dict_kwargs:
        merged.update(dict_kwargs)

    return merged


def _obj_to_dict(obj: Any) -> Any:
    """Convert an object to a dictionary representation for tracing.

    Args:
        obj (Any): The object to convert.

    Returns:
        Any: The dictionary representation of the object, or the object
            itself if it's a primitive type.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}  # obj
    elif isinstance(obj, (list, set, tuple)):
        return [_obj_to_dict(item) for item in obj]
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    else:
        result = None
        try:
            result = str(obj)
        except Exception as e:
            print(f"{obj} str method failed with error: {e}")
        return result
