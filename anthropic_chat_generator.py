"""Custom Anthropic Chat Generator for Haystack"""

import os
import json
from typing import List, Optional
from haystack import component
from haystack.dataclasses import ChatMessage
from anthropic import Anthropic


@component
class AnthropicChatGenerator:
    """
    A chat generator component that uses Anthropic's Claude API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", json_mode: bool = False):
        """
        Initialize the Anthropic Chat Generator.

        :param api_key: Anthropic API key. If not provided, will try to get from ANTHROPIC_API_KEY env var.
        :param model: The model to use (e.g., "claude-3-5-sonnet-20241022", "claude-sonnet-4-5-20250929")
        :param json_mode: If True, requests JSON-formatted responses. Default is False for plain text.
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.json_mode = json_mode

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        """
        Generate a reply using Anthropic's Claude API.

        :param messages: List of ChatMessage objects representing the conversation.
        :return: Dictionary with "replies" key containing a list of ChatMessage objects.
        """
        # Convert Haystack ChatMessage to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.is_from("user"):
                content = msg.text
                # If json_mode is enabled, add JSON instruction to the last user message
                if self.json_mode and msg == messages[-1]:
                    content = msg.text + "\n\nIMPORTANT: You must respond with valid JSON only. Do not include any text outside of the JSON structure."
                anthropic_messages.append({"role": "user", "content": content})
            elif msg.is_from("assistant"):
                anthropic_messages.append({"role": "assistant", "content": msg.text})
            elif msg.is_from("system"):
                # Store system message separately for Anthropic API
                system_message = msg.text
                if self.json_mode:
                    system_message = (system_message or "") + "\n\nYou must always respond with valid JSON format only."

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": anthropic_messages
        }
        
        # Add system message if present
        if system_message:
            api_params["system"] = system_message
        elif self.json_mode:
            # Add system instruction for JSON mode if no system message exists
            api_params["system"] = "You must always respond with valid JSON format only. Do not include any explanatory text outside the JSON structure."

        # Call Anthropic API
        response = self.client.messages.create(**api_params)

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

        # If json_mode is enabled, validate and clean the JSON response
        if self.json_mode:
            # Try to extract JSON from the response (in case there's extra text)
            reply_content = reply_content.strip()
            # Try to find JSON object in the response
            try:
                # First, try parsing the entire response as JSON
                json.loads(reply_content)
                # If successful, keep it as is
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the text
                # Look for JSON object boundaries
                start_idx = reply_content.find('{')
                end_idx = reply_content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = reply_content[start_idx:end_idx + 1]
                    try:
                        json.loads(json_str)
                        reply_content = json_str
                    except json.JSONDecodeError:
                        pass  # Keep original if extraction fails

        # Create a ChatMessage reply
        reply = ChatMessage.from_assistant(reply_content)
        
        return {"replies": [reply]}

