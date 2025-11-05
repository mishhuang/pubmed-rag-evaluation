"""Custom Anthropic Chat Generator for Haystack"""

import os
from typing import List, Optional
from haystack import component
from haystack.dataclasses import ChatMessage
from anthropic import Anthropic


@component
class AnthropicChatGenerator:
    """
    A chat generator component that uses Anthropic's Claude API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Anthropic Chat Generator.

        :param api_key: Anthropic API key. If not provided, will try to get from ANTHROPIC_API_KEY env var.
        :param model: The model to use (e.g., "claude-3-5-sonnet-20241022", "claude-sonnet-4-5-20250929")
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        """
        Generate a reply using Anthropic's Claude API.

        :param messages: List of ChatMessage objects representing the conversation.
        :return: Dictionary with "replies" key containing a list of ChatMessage objects.
        """
        # Convert Haystack ChatMessage to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg.is_from("user"):
                anthropic_messages.append({"role": "user", "content": msg.text})
            elif msg.is_from("assistant"):
                anthropic_messages.append({"role": "assistant", "content": msg.text})
            elif msg.is_from("system"):
                # Anthropic handles system messages differently
                anthropic_messages.append({"role": "user", "content": f"System: {msg.text}"})

        # Call Anthropic API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=anthropic_messages
        )

        # Extract the reply content
        reply_content = ""
        if response.content:
            # Anthropic returns content as a list of TextBlock objects
            for block in response.content:
                if hasattr(block, 'text'):
                    reply_content += block.text
                elif isinstance(block, dict) and 'text' in block:
                    reply_content += block['text']
                elif hasattr(block, 'type') and block.type == 'text':
                    reply_content += block.text
                else:
                    reply_content += str(block)

        # Create a ChatMessage reply
        reply = ChatMessage.from_assistant(reply_content)
        
        return {"replies": [reply]}

