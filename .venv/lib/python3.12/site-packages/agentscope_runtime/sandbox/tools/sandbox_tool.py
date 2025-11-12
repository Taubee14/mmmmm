# -*- coding: utf-8 -*-
from typing import Optional, Any, Dict

from .tool import Tool
from ..enums import SandboxType
from ..box.sandbox import Sandbox


class SandboxTool(Tool):
    """Built-in tool class"""

    _name: str = None
    _sandbox_type: SandboxType | str = None
    _tool_type: str = None
    _schema: Dict = None

    def __init__(
        self,
        *,
        sandbox: Optional[Any] = None,
        name: Optional[str] = None,
        sandbox_type: Optional[SandboxType | str] = None,
        tool_type: Optional[str] = None,
        schema: Optional[Dict] = None,
    ):
        """
        Initialize the tool.

        Note: Once the sandbox is set, it does not change.
        """
        self._sandbox = sandbox
        self._name = name or self.__class__._name
        self._sandbox_type = sandbox_type or self.__class__._sandbox_type
        self._tool_type = tool_type or self.__class__._tool_type
        self._schema = schema or self.__class__._schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def sandbox_type(self) -> SandboxType:
        return SandboxType(self._sandbox_type)

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
    def sandbox(self):
        return self._sandbox

    def __call__(self, *, sandbox: Optional[Any] = None, **kwargs):
        """Call the tool, allowing a temporary sandbox to be specified"""
        return self.call(sandbox=sandbox, **kwargs)

    def call(self, *, sandbox: Optional[Any] = None, **kwargs):
        """
        Execute the tool call.
        Args:
            sandbox: Temporarily used sandbox, highest priority
            **kwargs: Tool parameters
        """
        # Priority: temporary sandbox > instance sandbox > None (dryrun)
        box = sandbox or self._sandbox
        if box is None:
            return self._dryrun_call(**kwargs)
        else:
            return box.call_tool(self.name, arguments=kwargs)

    def bind(self, sandbox: Sandbox):
        """
        Return a new instance bound with a specific sandbox (immutable mode).
        """
        if not isinstance(sandbox, Sandbox):
            raise TypeError(
                "The provided sandbox must be an instance of `Sandbox`.",
            )

        assert self.sandbox_type == sandbox.sandbox_type, (
            f"Sandbox type mismatch! The tool requires a sandbox of type "
            f"`{self.sandbox_type}`, but a sandbox of type"
            f" `{sandbox.sandbox_type}` was provided."
        )

        return self.__class__(sandbox=sandbox)

    def _dryrun_call(self, **kwargs):
        """
        Dryrun mode: temporarily create a sandbox
        """
        from ..registry import SandboxRegistry

        cls_ = SandboxRegistry.get_classes_by_type(self.sandbox_type)

        with cls_() as box:
            return box.call_tool(self.name, arguments=kwargs)
