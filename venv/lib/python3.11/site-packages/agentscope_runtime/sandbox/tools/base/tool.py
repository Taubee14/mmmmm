# -*- coding: utf-8 -*-
from typing import Dict

from ..sandbox_tool import SandboxTool


class RunIPythonCellTool(SandboxTool):
    """Tool for running IPython cells."""

    _name: str = "run_ipython_cell"
    _sandbox_type: str = "base"
    _tool_type: str = "generic"
    _schema: Dict = {
        "name": "run_ipython_cell",
        "description": "Run an IPython cell.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "IPython code to execute",
                },
            },
            "required": ["code"],
        },
    }


class RunShellCommandTool(SandboxTool):
    """Tool for running shell commands."""

    _name: str = "run_shell_command"
    _sandbox_type: str = "base"
    _tool_type: str = "generic"
    _schema: Dict = {
        "name": "run_shell_command",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        },
    }


run_ipython_cell = RunIPythonCellTool()
run_shell_command = RunShellCommandTool()
