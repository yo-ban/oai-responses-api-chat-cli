# Purpose: Wrap the OpenAI Responses API client for chat interactions.
# Why: Encapsulation simplifies retries, parsing, and future enhancements.
"""API client helpers for talking to OpenAI's Responses endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, cast

from openai import OpenAI
from openai.types.responses import Response


@dataclass(frozen=True)
class ChatResult:
    """Normalized response payload returned to the CLI."""

    message_text: str
    response_id: str
    output_items: List[Dict[str, object]]


class ResponsesChatClient:
    """Thin wrapper that sends chat turns via the Responses API."""

    def __init__(
        self,
        *,
        base_url: Optional[str],
        api_key: str,
        model: str,
    ) -> None:
        """Configure the OpenAI client."""
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def send_message(
        self,
        *,
        context: Sequence[Dict[str, object]],
        developer_prompt: Optional[str],
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Send a message with stateless context management."""
        options: Dict[str, object] = {
            "model": self._model,
            "input": list(context),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if developer_prompt:
            options["instructions"] = developer_prompt
        if max_output_tokens is not None:
            options["max_output_tokens"] = max_output_tokens
        if reasoning:
            options["reasoning"] = reasoning

        response: Response = self._client.responses.create(
            **options,  # type: ignore[arg-type]
        )
        text = self._extract_text(response)
        raw_output = response.output or []
        serialized_output = self._sanitize_output_items(raw_output)
        return ChatResult(
            message_text=text,
            response_id=response.id,
            output_items=serialized_output,
        )

    @staticmethod
    def _extract_text(response: Response) -> str:
        """Extract the assistant message text from a response payload."""
        for output in response.output or []:
            if getattr(output, "type", None) != "message":
                continue
            for item in getattr(output, "content", []):
                if getattr(item, "type", None) == "output_text":
                    text_value = getattr(item, "text", None)
                    if text_value:
                        return text_value
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        raise ValueError("No assistant text found in response payload.")

    @staticmethod
    def _serialize_object(item: Any) -> Dict[str, object]:
        """Normalize output objects into dictionaries."""
        if hasattr(item, "model_dump"):
            return cast(Dict[str, object], item.model_dump())
        if isinstance(item, dict):
            return cast(Dict[str, object], item)
        raise TypeError(f"Unsupported output item type: {type(item)!r}")

    @classmethod
    def _sanitize_output_items(cls, output: Sequence[Any]) -> List[Dict[str, object]]:
        """Convert SDK output items to plain dictionaries without dropping fields."""
        serialized: List[Dict[str, object]] = []
        for item in output:
            normalized = cls._serialize_object(item)
            # Copy to avoid mutating SDK objects; retain all keys (including id/status).
            serialized.append(dict(normalized))
        return serialized
