"""
ActionAPIClient：封装 DashScope 兼容接口的聊天调用。

常见使用场景：

1. 完全自定义 prompt

    >>> from action_api_client import ActionAPIClient
    >>> client = ActionAPIClient(api_key="your-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    >>> resp = client.chat_completion([
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "Hello!"}
    ... ])
    >>> print(resp.choices[0].message.content)

2. 复用 generate_actions 便捷方法（可作为 Planner/Solver 的后端调用）

    >>> result = client.generate_actions(
    ...     prompt="请列出完成任务的三个步骤，并以 JSON 返回",
    ...     system_prompt="你是一个严谨的规划助手，只输出 JSON。"
    ... )
    >>> print(result["content"])

无需在类里写死 prompt：
- 实例化时不给 `system_prompt`，调用时临时传入即可；
- 或者直接使用 `chat_completion(messages=[...])` 完全自定义对话。
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

DEFAULT_MODEL_NAME = os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus")


class ActionAPIClient:
    """
    对接大模型聊天接口的轻量封装，便于在多处复用。

    - 统一管理 `OpenAI` 客户端、模型参数等基础配置。
    - 提供 `chat_completion` 让调用者完全自定义 messages。
    - 提供 `generate_actions` 作为默认实现，仍兼容原先的动作 JSON 输出需求。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required to initialize ActionAPIClient")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model or DEFAULT_MODEL_NAME
        self._default_system_prompt = system_prompt

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ):
        """
        直接调用底层的 chat.completions.create；使用者可完全自定义 prompt。
        返回 OpenAI SDK 的原始响应对象，方便自行解析。
        """
        return self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def generate_actions(
        self,
        prompt: str,
        rag_context: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        model: Optional[str] = None,
    ) -> dict:
        """
        让模型一次性产出一个 JSON 对象：包含 content 数组（若未完成，最后附一条 status=未完成）

        使用者仍然可以通过 chat_completion 自行构造其它 prompt，这里只是一个便利方法。
        """
        system_prompt = system_prompt or self._default_system_prompt
        user_prompt = prompt
        if rag_context:
            user_prompt += f"\n（历史/上下文）{rag_context}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
