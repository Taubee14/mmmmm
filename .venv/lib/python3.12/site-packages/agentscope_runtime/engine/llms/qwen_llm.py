# -*- coding: utf-8 -*-
import os

from openai import Client, AsyncClient

from .base_llm import BaseLLM


class QwenLLM(BaseLLM):
    """
    QwenLLM is a class that provides a wrapper around the Qwen LLM model.
    """

    base_url = None

    def __init__(
        self,
        model_name: str = "qwen-turbo",
        api_key: str = None,
        **kwargs,
    ):
        """
        Initialize the QwenLLM class.

        Args:
            model_name (str): The name of the Qwen LLM model to use.
                Defaults to "qwen-turbo".
            api_key (str): The API key for Qwen service.
                If None, will read from DASHSCOPE_API_KEY environment variable.
        """
        super().__init__(model_name, **kwargs)

        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if self.base_url is None:
            default_base_url = (
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.base_url = os.getenv("DASHSCOPE_BASE_URL", default_base_url)
        self.client = Client(
            api_key=api_key,
            base_url=self.base_url,
        )
        self.async_client = AsyncClient(
            api_key=api_key,
            base_url=self.base_url,
        )
