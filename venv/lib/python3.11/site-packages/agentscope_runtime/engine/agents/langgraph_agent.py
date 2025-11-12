# -*- coding: utf-8 -*-
import json

from langgraph.graph.state import CompiledStateGraph

from .base_agent import Agent
from ..schemas.agent_schemas import (
    Message,
    TextContent,
)


def _state_folder(messages):
    if len(messages) > 0:
        return json.loads(messages[0]["content"])
    else:
        return []


def _state_unfolder(state):
    state_jsons = json.dumps(state)
    return state_jsons


class LangGraphAgent(Agent):
    def __init__(
        self,
        graph: CompiledStateGraph = None,
        state_folder=_state_folder,
        state_unfolder=_state_unfolder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_folder = state_folder
        self.state_unfolder = state_unfolder
        self.graph = graph

    async def run_async(
        self,
        context,
        **kwargs,
    ):
        # fold the last m
        list_messages = []
        for m in context.session.messages:
            dumped = m.model_dump()
            dumped["content"] = dumped["content"][0]["text"]
            list_messages.append(dumped)

        _input = self.state_folder(list_messages)

        output = await self.graph.ainvoke(_input)
        content = self.state_unfolder(output)

        message = Message(role="assistant")
        text = TextContent(type="text", text=content)
        message.add_content(text)
        message.completed()
        yield message
