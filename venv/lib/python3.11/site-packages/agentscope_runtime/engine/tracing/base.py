# -*- coding: utf-8 -*-
# pylint: disable=protected-access
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List

from .tracing_metric import TraceType


# Handler Interface
class TracerHandler(ABC):
    """Abstract base class for tracer handlers."""

    @abstractmethod
    def on_start(
        self,
        event_type: TraceType,
        payload: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle the start of a trace event.

        Args:
            event_type (TraceType): The type of event being traced.
            payload (Dict[str, Any]): The payload data for the event.
            **kwargs (Any): Additional keyword arguments.
        """

    @abstractmethod
    def on_end(
        self,
        event_type: TraceType,
        start_payload: Dict[str, Any],
        end_payload: Dict[str, Any],
        start_time: float,
        **kwargs: Any,
    ) -> None:
        """Handle the end of a trace event.

        Args:
            event_type (TraceType): The type of event being traced.
            start_payload (Dict[str, Any]): The payload data from event start.
            end_payload (Dict[str, Any]): The payload data from event end.
            start_time (float): The timestamp when the event started.
            **kwargs (Any): Additional keyword arguments.
        """

    @abstractmethod
    def on_log(self, message: str, **kwargs: Any) -> None:
        """Handle a log message during tracing.

        Args:
            message (str): The log message.
            **kwargs (Any): Additional keyword arguments.
        """

    @abstractmethod
    def on_error(
        self,
        event_type: TraceType,
        start_payload: Dict[str, Any],
        error: Exception,
        start_time: float,
        traceback_info: str,
        **kwargs: Any,
    ) -> None:
        """Handle an error during tracing.

        Args:
            event_type (TraceType): The type of event being traced.
            start_payload (Dict[str, Any]): The payload data from event start.
            error (Exception): The exception that occurred.
            start_time (float): The timestamp when the event started.
            traceback_info (str): The traceback information.
            **kwargs (Any): Additional keyword arguments.
        """


class BaseLogHandler(TracerHandler):
    """Basic log handler implementation using Python's logging module."""

    import logging

    logger = logging.getLogger(__name__)

    def on_start(
        self,
        event_type: TraceType,
        payload: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Log the start of a trace event.

        Args:
            event_type (TraceType): The type of event being traced.
            payload (Dict[str, Any]): The payload data for the event.
            **kwargs (Any): Additional keyword arguments.
        """
        self.logger.info(f"Event {event_type} started with payload: {payload}")

    def on_end(
        self,
        event_type: TraceType,
        start_payload: Dict[str, Any],
        end_payload: Dict[str, Any],
        start_time: float,
        **kwargs: Any,
    ) -> None:
        """Log the end of a trace event.

        Args:
            event_type (TraceType): The type of event being traced.
            start_payload (Dict[str, Any]): The payload data from event start.
            end_payload (Dict[str, Any]): The payload data from event end.
            start_time (float): The timestamp when the event started.
            **kwargs (Any): Additional keyword arguments.
        """
        self.logger.info(
            f"Event {event_type} ended with start payload: {start_payload}, "
            f"end payload: {end_payload}, duration: "
            f"{time.time() - start_time} seconds, kwargs: {kwargs}",
        )

    def on_log(self, message: str, **kwargs: Any) -> None:
        """Log a message during tracing.

        Args:
            message (str): The log message.
            **kwargs (Any): Additional keyword arguments.
        """
        if message:
            self.logger.info(f"Log: {message}")

    def on_error(
        self,
        event_type: TraceType,
        start_payload: Dict[str, Any],
        error: Exception,
        start_time: float,
        traceback_info: str,
        **kwargs: Any,
    ) -> None:
        """Log an error during tracing.

        Args:
            event_type (TraceType): The type of event being traced.
            start_payload (Dict[str, Any]): The payload data from event start.
            error (Exception): The exception that occurred.
            start_time (float): The timestamp when the event started.
            traceback_info (str): The traceback information.
            **kwargs (Any): Additional keyword arguments.
        """
        self.logger.error(
            f"Error in event {event_type} with payload: {start_payload}, "
            f"error: {error}, "
            f"traceback: {traceback_info}, duration: "
            f"{time.time() - start_time} seconds, kwargs: {kwargs}",
        )


class Tracer:
    """
    Tracer class for logging events
    usage:
    with tracer.event(TraceType.LLM, payload) as event:
        event.log("message")
        ""...logic here...""
        end_payload = {xxx}
        # optional on_end call for additional payload and kwargs
        event.on_end(end_payload, if_success=True)
    """

    def __init__(self, handlers: List[TracerHandler]):
        """Initialize the tracer with a list of handlers.

        Args:
            handlers (List[TracerHandler]): List of handlers to process trace
            events.
        """
        self.handlers = handlers

    @contextmanager
    def event(
        self,
        event_type: TraceType,
        payload: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Create a context manager for tracing an event.

        Args:
            event_type (TraceType): The type of event being traced.
            payload (Dict[str, Any]): The payload data for the event.
            **kwargs (Any): Additional keyword arguments.

        Yields:
            EventContext: The event context for logging and managing the trace.
        """
        start_time = time.time()

        for handle in self.handlers:
            handle.on_start(event_type, payload, **kwargs)

        event_context = EventContext(
            self.handlers,
            event_type,
            start_time,
            payload,
        )
        try:
            yield event_context
        except Exception as e:
            traceback_info = traceback.format_exc()
            for handle in self.handlers:
                handle.on_error(
                    event_type,
                    payload,
                    e,
                    start_time,
                    traceback_info=traceback_info,
                )
            raise
        event_context._end(payload)

    def log(self, message: str, **kwargs: Any) -> None:
        """Log a message using all registered handlers.

        Args:
            message (str): The log message.
            **kwargs (Any): Additional keyword arguments.
        """
        for handle in self.handlers:
            handle.on_log(message, **kwargs)


class EventContext:
    """Context manager for individual trace events."""

    def __init__(
        self,
        handlers: List[TracerHandler],
        event_type: TraceType,
        start_time: float,
        start_payload: Dict[str, Any],
    ) -> None:
        """Initialize the event context.

        Args:
            handlers (List[TracerHandler]): List of handlers to process
            trace  events.
            event_type (TraceType): The type of event being traced.
            start_time (float): The timestamp when the event started.
            start_payload (Dict[str, Any]): The payload data from event start.
        """
        self.handlers = handlers
        self.event_type = event_type
        self.start_time = start_time
        self.start_payload = start_payload
        self.end_payload = {}
        self.kwargs = {}

    def on_end(self, payload: Dict[str, Any], **kwargs: Any) -> None:
        """Set the end payload and additional kwargs for the event.

        Args:
            payload (Dict[str, Any]): The payload data for event end.
            **kwargs (Any): Additional keyword arguments.
        """
        self.end_payload = payload
        self.kwargs = kwargs

    def on_log(self, message: str, **kwargs: Any) -> None:
        """Log a message within this event context.

        Args:
            message (str): The log message.
            **kwargs (Any): Additional keyword arguments.
        """
        kwargs["event_type"] = self.event_type
        kwargs["start_time"] = self.start_time
        kwargs["start_payload"] = self.start_payload
        for handle in self.handlers:
            handle.on_log(message, **kwargs)

    def _end(self, start_payload: Dict[str, Any] = None) -> None:
        """Finalize the event by calling on_end for all handlers.

        Args:
            start_payload (Dict[str, Any], optional): The payload data from
            event start.
        """
        for handle in self.handlers:
            handle.on_end(
                self.event_type,
                start_payload,
                self.end_payload,
                self.start_time,
                **self.kwargs,
            )

    def set_trace_header(self, trace_header: Dict[str, Any]) -> None:
        """Set trace header for compatible handlers.

        Args:
            trace_header (Dict[str, Any]): The trace header information.
        """
        # TODO: Implement this
