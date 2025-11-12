# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
from typing import Dict

from ..base.tool import RunIPythonCellTool, RunShellCommandTool
from ..sandbox_tool import SandboxTool


class _RunIPythonCellTool(RunIPythonCellTool):
    """Tool for running IPython cells."""

    _sandbox_type: str = "filesystem"
    _tool_type: str = "generic"


class _RunShellCommandTool(RunShellCommandTool):
    """Tool for running shell commands."""

    _sandbox_type: str = "filesystem"
    _tool_type: str = "generic"


class ReadFileTool(SandboxTool):
    """Tool for reading the complete contents of a file."""

    _name: str = "read_file"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "read_file",
        "description": "Read the complete contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["path"],
        },
    }


class ReadMultipleFilesTool(SandboxTool):
    """Tool for reading the contents of multiple files."""

    _name: str = "read_multiple_files"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "read_multiple_files",
        "description": "Read the contents of multiple files simultaneously.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": "Paths to the files to read",
                },
            },
            "required": ["paths"],
        },
    }


class WriteFileTool(SandboxTool):
    """Tool for creating or overwriting a file with new content."""

    _name: str = "write_file"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "write_file",
        "description": "Create a new file or overwrite an existing file with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write to",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write into the file",
                },
            },
            "required": ["path", "content"],
        },
    }


class EditFileTool(SandboxTool):
    """Tool for making line-based edits to a text file."""

    _name: str = "edit_file"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "edit_file",
        "description": "Make line-based edits to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Text to search for - must match exactly",
                            },
                            "newText": {
                                "type": "string",
                                "description": "Text to replace with",
                            },
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": False,
                    },
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "Preview changes using git-style diff format",
                },
            },
            "required": ["path", "edits"],
        },
    }


class CreateDirectoryTool(SandboxTool):
    """Tool for creating a new directory or ensuring it exists."""

    _name: str = "create_directory"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "create_directory",
        "description": "Create a new directory or ensure a directory exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to create",
                },
            },
            "required": ["path"],
        },
    }


class ListDirectoryTool(SandboxTool):
    """Tool for listing all files and directories in a specified path."""

    _name: str = "list_directory"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "list_directory",
        "description": "Get a detailed listing of all files and directories in a specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list contents",
                },
            },
            "required": ["path"],
        },
    }


class DirectoryTreeTool(SandboxTool):
    """Tool for getting a recursive tree view of files and directories."""

    _name: str = "directory_tree"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "directory_tree",
        "description": "Get a recursive tree view of files and directories as a JSON structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to get tree structure",
                },
            },
            "required": ["path"],
        },
    }


class MoveFileTool(SandboxTool):
    """Tool for moving or renaming files and directories."""

    _name: str = "move_file"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "move_file",
        "description": "Move or rename files and directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source path to move from",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path to move to",
                },
            },
            "required": ["source", "destination"],
        },
    }


class SearchFilesTool(SandboxTool):
    """Tool for searching files and directories matching a pattern."""

    _name: str = "search_files"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "search_files",
        "description": "Recursively search for files and directories matching a pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Starting path for the search",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to match files/directories",
                },
                "excludePatterns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "default": [],
                    "description": "Patterns to exclude from search",
                },
            },
            "required": ["path", "pattern"],
        },
    }


class GetFileInfoTool(SandboxTool):
    """Tool for retrieving metadata about a file or directory."""

    _name: str = "get_file_info"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "get_file_info",
        "description": "Retrieve detailed metadata about a file or directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
            },
            "required": ["path"],
        },
    }


class ListAllowedDirectoriesTool(SandboxTool):
    """Tool for listing allowed directories the server can access."""

    _name: str = "list_allowed_directories"
    _sandbox_type: str = "filesystem"
    _tool_type: str = "filesystem"
    _schema: Dict = {
        "name": "list_allowed_directories",
        "description": "Returns the list of directories that this server is allowed to access.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


run_shell_command = _RunShellCommandTool()
run_ipython_cell = _RunIPythonCellTool()
read_file = ReadFileTool()
read_multiple_files = ReadMultipleFilesTool()
write_file = WriteFileTool()
edit_file = EditFileTool()
create_directory = CreateDirectoryTool()
list_directory = ListDirectoryTool()
directory_tree = DirectoryTreeTool()
move_file = MoveFileTool()
search_files = SearchFilesTool()
get_file_info = GetFileInfoTool()
list_allowed_directories = ListAllowedDirectoriesTool()
