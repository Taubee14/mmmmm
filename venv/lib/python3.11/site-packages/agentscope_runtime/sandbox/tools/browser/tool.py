# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long
from typing import Dict

from ..base.tool import RunIPythonCellTool, RunShellCommandTool
from ..sandbox_tool import SandboxTool


class _RunIPythonCellTool(RunIPythonCellTool):
    """Tool for running IPython cells."""

    _sandbox_type: str = "browser"
    _tool_type: str = "generic"


class _RunShellCommandTool(RunShellCommandTool):
    """Tool for running shell commands."""

    _sandbox_type: str = "browser"
    _tool_type: str = "generic"


class BrowserCloseTool(SandboxTool):
    """Tool for closing the browser page."""

    _name: str = "browser_close"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_close",
        "description": "Close the page",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserResizeTool(SandboxTool):
    """Tool for resizing the browser window."""

    _name: str = "browser_resize"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_resize",
        "description": "Resize the browser window",
        "parameters": {
            "type": "object",
            "properties": {
                "width": {
                    "type": "number",
                    "description": "Width of the browser window",
                },
                "height": {
                    "type": "number",
                    "description": "Height of the browser window",
                },
            },
            "required": ["width", "height"],
        },
    }


class BrowserConsoleMessagesTool(SandboxTool):
    """Tool for returning all console messages."""

    _name: str = "browser_console_messages"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_console_messages",
        "description": "Returns all console messages",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserHandleDialogTool(SandboxTool):
    """Tool for handling a dialog."""

    _name: str = "browser_handle_dialog"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_handle_dialog",
        "description": "Handle a dialog",
        "parameters": {
            "type": "object",
            "properties": {
                "accept": {
                    "type": "boolean",
                    "description": "Whether to accept the dialog.",
                },
                "promptText": {
                    "type": "string",
                    "description": "The text of the prompt in case of a prompt dialog.",
                },
            },
            "required": ["accept"],
        },
    }


class BrowserFileUploadTool(SandboxTool):
    """Tool for uploading one or multiple files."""

    _name: str = "browser_file_upload"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_file_upload",
        "description": "Upload one or multiple files",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": "The absolute paths to the files to upload. Can be a single file or multiple files.",
                },
            },
            "required": ["paths"],
        },
    }


class BrowserPressKeyTool(SandboxTool):
    """Tool for pressing a key on the keyboard."""

    _name: str = "browser_press_key"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_press_key",
        "description": "Press a key on the keyboard",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Name of the key to press or a character to generate, such as `ArrowLeft` or `a`",
                },
            },
            "required": ["key"],
        },
    }


class BrowserNavigateTool(SandboxTool):
    """Tool for navigating to a URL."""

    _name: str = "browser_navigate"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_navigate",
        "description": "Navigate to a URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to",
                },
            },
            "required": ["url"],
        },
    }


class BrowserNavigateBackTool(SandboxTool):
    """Tool for going back to the previous page."""

    _name: str = "browser_navigate_back"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_navigate_back",
        "description": "Go back to the previous page",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserNavigateForwardTool(SandboxTool):
    """Tool for going forward to the next page."""

    _name: str = "browser_navigate_forward"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_navigate_forward",
        "description": "Go forward to the next page",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserNetworkRequestsTool(SandboxTool):
    """Tool for returning all network requests since loading the page."""

    _name: str = "browser_network_requests"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_network_requests",
        "description": "Returns all network requests since loading the page",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserPdfSaveTool(SandboxTool):
    """Tool for saving a page as PDF."""

    _name: str = "browser_pdf_save"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_pdf_save",
        "description": "Save page as PDF",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File name to save the pdf to. Defaults to `page-{timestamp}.pdf` if not specified.",
                },
            },
            "required": [],
        },
    }


class BrowserTakeScreenshotTool(SandboxTool):
    """Tool for taking a screenshot of the current page."""

    _name: str = "browser_take_screenshot"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_take_screenshot",
        "description": "Take a screenshot of the current page. You can't perform actions based on the screenshot, use browser_snapshot for actions.",
        "parameters": {
            "type": "object",
            "properties": {
                "raw": {
                    "type": "boolean",
                    "description": "Whether to return without compression (in PNG format). Default is false, which returns a JPEG image.",
                },
                "filename": {
                    "type": "string",
                    "description": "File name to save the screenshot to. Defaults to `page-{timestamp}.{png|jpeg}` if not specified.",
                },
                "element": {
                    "type": "string",
                    "description": "Human-readable element description used to obtain permission to screenshot the element. If not provided, the screenshot will be taken of viewport. If element is provided, ref must be provided too.",
                },
                "ref": {
                    "type": "string",
                    "description": "Exact target element reference from the page snapshot. If not provided, the screenshot will be taken of viewport. If ref is provided, element must be provided too.",
                },
            },
            "required": [],
        },
    }


class BrowserSnapshotTool(SandboxTool):
    """Tool for capturing an accessibility snapshot of the current page."""

    _name: str = "browser_snapshot"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_snapshot",
        "description": "Capture accessibility snapshot of the current page, this is better than screenshot",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserClickTool(SandboxTool):
    """Tool for performing a click on a web page."""

    _name: str = "browser_click"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_click",
        "description": "Perform click on a web page",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Human-readable element description used to obtain permission to interact with the element",
                },
                "ref": {
                    "type": "string",
                    "description": "Exact target element reference from the page snapshot",
                },
            },
            "required": ["element", "ref"],
        },
    }


class BrowserDragTool(SandboxTool):
    """Tool for performing drag and drop between two elements."""

    _name: str = "browser_drag"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_drag",
        "description": "Perform drag and drop between two elements",
        "parameters": {
            "type": "object",
            "properties": {
                "startElement": {
                    "type": "string",
                    "description": "Human-readable source element description used to obtain the permission to interact with the element",
                },
                "startRef": {
                    "type": "string",
                    "description": "Exact source element reference from the page snapshot",
                },
                "endElement": {
                    "type": "string",
                    "description": "Human-readable target element description used to obtain the permission to interact with the element",
                },
                "endRef": {
                    "type": "string",
                    "description": "Exact target element reference from the page snapshot",
                },
            },
            "required": ["startElement", "startRef", "endElement", "endRef"],
        },
    }


class BrowserHoverTool(SandboxTool):
    """Tool for hovering over an element on a page."""

    _name: str = "browser_hover"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_hover",
        "description": "Hover over an element on a page",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Human-readable element description used to obtain permission to interact with the element",
                },
                "ref": {
                    "type": "string",
                    "description": "Exact target element reference from the page snapshot",
                },
            },
            "required": ["element", "ref"],
        },
    }


class BrowserTypeTool(SandboxTool):
    """Tool for typing text into an editable element."""

    _name: str = "browser_type"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_type",
        "description": "Type text into an editable element",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Human-readable element description used to obtain permission to interact with the element",
                },
                "ref": {
                    "type": "string",
                    "description": "Exact target element reference from the page snapshot",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the element",
                },
                "submit": {
                    "type": "boolean",
                    "description": "Whether to submit entered text (press Enter after)",
                },
                "slowly": {
                    "type": "boolean",
                    "description": "Whether to type one character at a time. Useful for triggering key handlers in the page. By default, the entire text is filled in at once.",
                },
            },
            "required": ["element", "ref", "text"],
        },
    }


class BrowserSelectOptionTool(SandboxTool):
    """Tool for selecting an option in a dropdown."""

    _name: str = "browser_select_option"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_select_option",
        "description": "Select an option in a dropdown",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Human-readable element description used to obtain permission to interact with the element",
                },
                "ref": {
                    "type": "string",
                    "description": "Exact target element reference from the page snapshot",
                },
                "values": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": "Array of values to select in the dropdown. This can be a single value or multiple values.",
                },
            },
            "required": ["element", "ref", "values"],
        },
    }


class BrowserTabListTool(SandboxTool):
    """Tool for listing browser tabs."""

    _name: str = "browser_tab_list"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_tab_list",
        "description": "List browser tabs",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


class BrowserTabNewTool(SandboxTool):
    """Tool for opening a new tab."""

    _name: str = "browser_tab_new"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_tab_new",
        "description": "Open a new tab",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to in the new tab. If not provided, the new tab will be blank.",
                },
            },
            "required": [],
        },
    }


class BrowserTabSelectTool(SandboxTool):
    """Tool for selecting a tab by index."""

    _name: str = "browser_tab_select"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_tab_select",
        "description": "Select a tab by index",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "number",
                    "description": "The index of the tab to select",
                },
            },
            "required": ["index"],
        },
    }


class BrowserTabCloseTool(SandboxTool):
    """Tool for closing a tab."""

    _name: str = "browser_tab_close"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_tab_close",
        "description": "Close a tab",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "number",
                    "description": "The index of the tab to close. Closes current tab if not provided.",
                },
            },
            "required": [],
        },
    }


class BrowserWaitForTool(SandboxTool):
    """Tool for waiting for text to appear or disappear or for a specified time to pass."""

    _name: str = "browser_wait_for"
    _sandbox_type: str = "browser"
    _tool_type: str = "playwright"
    _schema: Dict = {
        "name": "browser_wait_for",
        "description": "Wait for text to appear or disappear or a specified time to pass",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "number",
                    "description": "The time to wait in seconds",
                },
                "text": {
                    "type": "string",
                    "description": "The text to wait for",
                },
                "textGone": {
                    "type": "string",
                    "description": "The text to wait for to disappear",
                },
            },
            "required": [],
        },
    }


run_shell_command = _RunShellCommandTool()
run_ipython_cell = _RunIPythonCellTool()
browser_close = BrowserCloseTool()
browser_resize = BrowserResizeTool()
browser_console_messages = BrowserConsoleMessagesTool()
browser_handle_dialog = BrowserHandleDialogTool()
browser_file_upload = BrowserFileUploadTool()
browser_press_key = BrowserPressKeyTool()
browser_navigate = BrowserNavigateTool()
browser_navigate_back = BrowserNavigateBackTool()
browser_navigate_forward = BrowserNavigateForwardTool()
browser_network_requests = BrowserNetworkRequestsTool()
browser_pdf_save = BrowserPdfSaveTool()
browser_take_screenshot = BrowserTakeScreenshotTool()
browser_snapshot = BrowserSnapshotTool()
browser_click = BrowserClickTool()
browser_drag = BrowserDragTool()
browser_hover = BrowserHoverTool()
browser_type = BrowserTypeTool()
browser_select_option = BrowserSelectOptionTool()
browser_tab_list = BrowserTabListTool()
browser_tab_new = BrowserTabNewTool()
browser_tab_select = BrowserTabSelectTool()
browser_tab_close = BrowserTabCloseTool()
browser_wait_for = BrowserWaitForTool()
