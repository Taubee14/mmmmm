# -*- coding: utf-8 -*-
from typing import Callable

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from .a2a_agent_adapter import A2AExecutor
from ..protocol_adapter import ProtocolAdapter
from ....agents import Agent


class A2AFastAPIDefaultAdapter(ProtocolAdapter):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent

    def add_endpoint(self, app, func: Callable, **kwargs):
        request_handler = DefaultRequestHandler(
            agent_executor=A2AExecutor(func=func),
            task_store=InMemoryTaskStore(),
        )

        agent_card = self.get_agent_card(self._agent)

        server = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        server.add_routes_to_app(app)

    def get_agent_card(self, agent: Agent) -> AgentCard:
        capabilities = AgentCapabilities(
            streaming=False,
            push_notifications=False,
        )
        skill = AgentSkill(
            id="dialog",
            name="Natural Language Dialog Skill",
            description="Enables natural language conversation and dialogue "
            "with users",
            tags=["natural language", "dialog", "conversation"],
            examples=[
                "Hello, how are you?",
                "Can you help me with something?",
            ],
        )

        return AgentCard(
            capabilities=capabilities,
            skills=[skill],
            name=agent.name,
            description=agent.description,
            default_input_modes=["text"],
            default_output_modes=["text"],
            url="http://127.0.0.1:8090/",
            version="1.0.0",
        )
