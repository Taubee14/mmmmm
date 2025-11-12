# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
import traceback
from typing import Optional, Any, Dict, Callable, Union

import inspect
from functools import partial

from .tool import Tool
from ..registry import SandboxType


class FunctionTool(Tool):
    """Function tool class for direct function calls.

    This tool class is designed for calling regular Python functions directly,
    WITHOUT running in a sandbox environment. Unlike SandboxTool which executes
    tools within isolated sandbox environments, FunctionTool executes functions
    in the current Python process context.

    Key differences from SandboxTool:
    - No sandbox isolation: functions run in the same process.
    - Direct function execution with immediate access to local variables and
        imports.
    - No security boundaries: functions have full access to the runtime
        environment.
    - Suitable for trusted, lightweight operations.

    Use cases:
    - Simple computational functions.
    - Data processing utilities.
    - Internal helper functions.
    - Functions that don't require sandbox isolation.

    Security note:
    Since functions run without sandbox protection, ensure that:
    - Functions are from trusted sources.
    - Input validation is handled appropriately.
    - Functions don't perform dangerous operations in production.

    Example:
        def add_numbers(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(add_numbers)
        result = tool(a=1, b=2)  # Executes directly, returns 3
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        schema: Optional[Dict] = None,
        tool_type: str = "function",
        description: Optional[str] = None,
    ):
        """
        Initialize the function tool.
        Args:
            func: The function to call (can be a partial function)
            name: Tool name, defaults to function name
            schema: Tool schema, auto-generated if not provided
            tool_type: Tool type
            description: Tool description
        """
        self._func = func
        self._name: str = name or self._extract_function_name(func)
        self._tool_type: str = tool_type
        self._description = description or func.__doc__ or f"Call {self._name}"
        # Auto-generate schema if not provided
        if schema is None:
            self._schema = self._generate_schema_from_func(func)
        else:
            self._schema = {
                "name": self._name,
                "description": self._description,
                **schema,
            }

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_type(self) -> str:
        return self._tool_type

    @property
    def schema(self) -> Dict:
        return {
            "type": "function",
            "function": self._schema,
        }

    @property
    def sandbox_type(self) -> SandboxType:
        """Function tools don't need a sandbox type"""
        return SandboxType.DUMMY

    @property
    def sandbox(self) -> None:
        """Function tools don't have sandbox"""
        return None

    def __call__(self, **kwargs):
        """Call the function directly"""
        return self.call(**kwargs)

    def call(self, *, sandbox: Optional[Any] = None, **kwargs):
        """
        Execute the function call.
        Args:
            sandbox: Ignored for function tools (for interface compatibility)
            **kwargs: Function parameters
        """
        # Filter kwargs to only include parameters that the function accepts
        filtered_kwargs = self._filter_kwargs(kwargs)
        try:
            return {
                "meta": None,
                "content": [
                    {
                        "type": "text",
                        "text": str(self._func(**filtered_kwargs)),
                        "annotations": None,
                        "description": "None",
                    },
                ],
                "isError": False,
            }
        except Exception as e:
            return {
                "meta": None,
                "content": [
                    {
                        "type": "text",
                        "text": f"{e}:\n{traceback.format_exc()}",
                        "annotations": None,
                        "description": "None",
                    },
                ],
                "isError": True,
            }

    def bind(self, *args, **kwargs):
        """
        Return a new instance with pre-bound parameters (similar to
        functools.partial).
        """
        return self.__class__(
            func=self._func,
            name=self._name,
            tool_type=self._tool_type,
            description=self._description,
        )

    def _extract_function_name(self, func: Callable) -> str:
        """Extract function name from function or partial"""
        if isinstance(func, partial):
            return getattr(func.func, "__name__", "partial_function")
        return getattr(func, "__name__", "unknown_function")

    def _filter_kwargs(self, kwargs: Dict) -> Dict:
        """Filter kwargs to match function signature"""
        if isinstance(self._func, partial):
            sig = inspect.signature(self._func.func)
            # Remove already bound parameters
            bound_params = set(self._func.keywords.keys())
        else:
            sig = inspect.signature(self._func)
            bound_params = set()

        filtered = {}
        for param_name, param in sig.parameters.items():
            if param_name in bound_params:
                continue
            if param_name in kwargs:
                filtered[param_name] = kwargs[param_name]
            elif (
                param.default == inspect.Parameter.empty
                and param.kind != param.VAR_KEYWORD
            ):
                # Required parameter is missing
                raise TypeError(f"Missing required parameter: {param_name}")

        # If function accepts **kwargs, include remaining parameters
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            used_params = set(sig.parameters.keys()) | bound_params
            for key, value in kwargs.items():
                if key not in used_params:
                    filtered[key] = value

        return filtered

    def _generate_schema_from_func(self, func: Callable) -> Dict:
        """Generate schema from function signature"""
        if isinstance(func, partial):
            actual_func = func.func
            bound_kwargs = func.keywords
        else:
            actual_func = func
            bound_kwargs = {}

        sig = inspect.signature(actual_func)
        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip already bound parameters
            if param_name in bound_kwargs:
                continue

            param_info = self._get_param_info(param)
            parameters[param_name] = param_info

            # Check if parameter is required
            if (
                param.default == inspect.Parameter.empty
                and param.kind
                not in (
                    param.VAR_POSITIONAL,
                    param.VAR_KEYWORD,
                )
            ):
                required.append(param_name)

        schema = {
            "name": self._name,
            "description": self._description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        }

        return schema

    def _get_param_info(self, param: inspect.Parameter) -> Dict:
        """Get parameter info for schema"""
        param_info = {"type": "string"}  # default type

        # Try to infer type from annotation
        if param.annotation != inspect.Parameter.empty:
            param_info.update(
                self._annotation_to_schema_type(param.annotation),
            )

        # Add default value if exists
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default

        return param_info

    def _annotation_to_schema_type(self, annotation) -> Dict:
        """Convert Python type annotation to JSON schema type"""
        type_mapping = {
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            str: {"type": "string"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        # Handle Union types (Optional)
        if hasattr(annotation, "__origin__"):
            if annotation.__origin__ is Union:
                args = annotation.__args__
                if (
                    len(args) == 2 and type(None) in args
                ):  # This is Optional[T]
                    non_none_type = next(
                        arg for arg in args if arg is not type(None)
                    )
                    result = self._annotation_to_schema_type(non_none_type)
                    # Mark as optional (though this is handled in required
                    # array)
                    return result

        return type_mapping.get(annotation, {"type": "string"})


def create_function_tool(
    func: Callable,
    name: Optional[str] = None,
    **kwargs,
) -> FunctionTool:
    """
    Factory function to create a FunctionTool.

    Args:func: Function or partial function
        name: Tool name
        **kwargs: Additional arguments for FunctionToolReturns:
        FunctionTool instance
    """
    return FunctionTool(func=func, name=name, **kwargs)


def function_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **tool_kwargs,
):
    """
    Decorator to convert a function into a FunctionTool.
    Usage:
        @function_tool(name="my_calculator", description="A simple calculator")
        def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(func):
        return FunctionTool(
            func=func,
            name=name,
            description=description,
            **tool_kwargs,
        )

    return decorator
