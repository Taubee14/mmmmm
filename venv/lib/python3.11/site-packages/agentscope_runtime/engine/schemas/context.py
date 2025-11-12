# -*- coding: utf-8 -*-
from typing import Optional, List, Dict

from pydantic import BaseModel, ConfigDict

from .agent_schemas import AgentRequest
from .agent_schemas import Message
from ..agents.base_agent import Agent
from ..services.context_manager import ContextManager

from ..services.session_history_service import (
    Session,
)
from ..services.environment_manager import EnvironmentManager


class Context(BaseModel):
    """
    Holds all contextual information for a single agent invocation.

    This object is created by the Runner and passed through the agent
    execution flow, providing access to necessary services and data,
    including a live request queue for real-time interaction.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    # Core context
    user_id: str
    session: Session = Session(id="", user_id="")
    activate_tools: list = []
    new_message: Optional[Message] = None
    current_messages: List[Message] = []
    request: AgentRequest
    new_message_dict: Optional[Dict] = None
    messages_list: List[Dict] = []

    # Services available to the agent

    environment_manager: Optional[EnvironmentManager] = None
    context_manager: Optional[ContextManager] = None
    # Agent specific config
    agent: Agent
    agent_config: Optional[dict] = None

    @property
    def messages(self):
        if self.new_message_dict:
            return self.messages_list + [self.new_message_dict]
        else:
            return self.messages_list
