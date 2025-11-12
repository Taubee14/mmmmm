# -*- coding: utf-8 -*-
def setup_tools(
    tools,
    environment_manager,
    session_id,
    user_id,
    include_schemas=False,
):
    """
    Generic function to activate tools.

    Args:
        tools: List of tools to activate.
        environment_manager: Environment manager
        session_id: ID of the session.
        user_id: ID of the user.
        include_schemas: Boolean flag to indicate if schemas should be
            included.

    Returns:
        tuple: (activated_tools, schemas) if include_schemas=True,
            else only activated_tools.
    """
    # Lazy import
    from .tool import Tool
    from .sandbox_tool import SandboxTool
    from ...sandbox.box.dummy.dummy_sandbox import DummySandbox

    enable_sandbox = False

    # Check tool types and determine if sandbox needs to be enabled
    for tool in tools:
        assert isinstance(tool, Tool), f"{tool} must be an instance of Tool"
        if isinstance(tool, SandboxTool):
            enable_sandbox = True
            break

    # Check environment service
    if enable_sandbox and environment_manager is None:
        raise ValueError("environment_manager is not set")

    # Connect to sandbox if required
    if enable_sandbox:
        sandboxes = environment_manager.connect_sandbox(
            session_id=session_id,
            user_id=user_id,
            tools=tools,
        )
    else:
        sandboxes = [DummySandbox()]

    # Bind tools to sandbox and prepare schemas if required
    schemas = []  # Initialize schemas list
    activated_tools = []  # Initialize activated tools list

    for tool in tools:
        if include_schemas:
            schemas.append(tool.schema)  # Collect tool schemas
        for sandbox in sandboxes:
            if sandbox.sandbox_type == tool.sandbox_type:
                bound_tool = tool.bind(sandbox)  # Bind tool to the sandbox
                activated_tools.append(
                    bound_tool,
                )  # Add to the activated tools list

    if include_schemas:
        return activated_tools, schemas
    return activated_tools
