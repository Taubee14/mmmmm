# -*- coding: utf-8 -*-

from .base_agent import Agent
from ..llms import BaseLLM
from ..schemas.agent_schemas import (
    Message,
    TextContent,
    convert_to_openai_messages,
    MessageType,
    convert_to_openai_tools,
)


class LLMAgent(Agent):
    def __init__(
        self,
        model: BaseLLM,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.model = model

    async def run_async(
        self,
        context,
        **kwargs,
    ):
        # agent request --> model request
        openai_messages = convert_to_openai_messages(context.session.messages)
        tools = convert_to_openai_tools(context.request.tools)

        # Step 3: Create initial Message
        message = Message(type=MessageType.MESSAGE, role="assistant")
        yield message.in_progress()

        # Step 4: LLM Content delta
        text_delta_content = TextContent(delta=True)
        async for chunk in self.model.chat_stream(openai_messages, tools):
            delta = chunk.choices[0].delta

            if delta.content:
                text_delta_content.text = delta.content
                text_delta_content = message.add_delta_content(
                    new_content=text_delta_content,
                )
                yield text_delta_content

        message.completed()
        yield message
