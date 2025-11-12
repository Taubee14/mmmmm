# -*- coding: utf-8 -*-
from typing import AsyncGenerator

from ..schemas.agent_schemas import Event


class Agent:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        before_agent_callback=None,
        after_agent_callback=None,
        agent_config=None,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.before_agent_callback = before_agent_callback or []
        self.after_agent_callback = after_agent_callback or []
        self.agent_config = agent_config or {}
        self.kwargs = kwargs

    async def run_async(
        self,
        context,
        **kwargs,
    ) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Subclasses must implement this method")
