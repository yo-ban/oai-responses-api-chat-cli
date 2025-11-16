# Purpose: Provide utilities for persisting and retrieving chat conversation history.
# Why: Isolating persistence logic enables easier testing and future format changes.
"""YAML-backed chat history management with resumable response ids."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from uuid import uuid4

import yaml

from .assets import encode_asset


class LiteralString(str):
    """Marker type to ensure YAML uses literal block style for multiline strings."""


def _literal_str_representer(dumper: yaml.SafeDumper, data: LiteralString) -> yaml.ScalarNode:
    """Represent LiteralString with literal block scalar style."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.SafeDumper.add_representer(LiteralString, _literal_str_representer)


@dataclass(frozen=True)
class HistoryEntry:
    """Represent a single message stored in the history log."""

    message_id: str
    role: str
    content: str = ""
    timestamp: str = ""
    response_id: Optional[str] = None
    input: Optional[List[Dict[str, object]]] = None
    output: Optional[List[Dict[str, object]]] = None


class ChatHistory:
    """Manage chat history persisted as a human-editable YAML document."""

    def __init__(self, path: Path) -> None:
        """Initialize the history manager and ensure the backing file exists."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Perform migrations when a history file already exists."""
        if self._path.exists():
            self._migrate_if_needed()

    @property
    def path(self) -> Path:
        """Return the filesystem path for the history file."""
        return self._path

    def iter_entries(self) -> Iterator[HistoryEntry]:
        """Yield history entries in chronological order."""
        for entry in self._read_entries():
            yield entry

    def last_response_id(self) -> Optional[str]:
        """Return the most recent assistant response identifier if present."""
        entries = list(self.iter_entries())
        for entry in reversed(entries):
            if entry.role == "assistant" and entry.response_id:
                return entry.response_id
        return None

    def clear(self) -> None:
        """Reset the history file to an empty state."""
        if self._path.exists():
            self._path.unlink()

    def entries(self) -> List[HistoryEntry]:
        """Return all stored history entries."""
        return self._read_entries()

    def record_turn(
        self,
        user_content: str,
        assistant_content: str,
        response_id: str,
        input_blocks: List[Dict[str, object]],
        output: Optional[List[Dict[str, object]]],
    ) -> None:
        """Persist a successful user/assistant turn."""
        sanitized_output = self._sanitize_output(output)
        user_display = self._display_from_input(input_blocks)
        assistant_display = self._display_from_output(sanitized_output) or assistant_content
        entries = self._read_entries()
        entries.extend(
            [
                self._build_entry(
                    role="user",
                    content=user_display,
                    input_blocks=input_blocks,
                ),
                self._build_entry(
                    role="assistant",
                    content=assistant_display,
                    response_id=response_id,
                    output=sanitized_output,
                ),
            ]
        )
        self._write_entries(entries)

    def _read_entries(self) -> List[HistoryEntry]:
        """Return all entries currently stored on disk."""
        if not self._path.exists():
            return []
        raw_text = self._path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []
        data = yaml.safe_load(raw_text)
        if not data:
            return []
        entries: List[HistoryEntry] = []
        for item in data:
            input_blocks = item.get("input")
            raw_output = item.get("output")
            sanitized_output = self._sanitize_output(raw_output)
            content_value = item.get("content")
            if content_value is None:
                if item["role"] == "user":
                    content_value = self._display_from_input(input_blocks)
                elif item["role"] == "assistant":
                    content_value = self._display_from_output(sanitized_output)
                else:
                    content_value = ""
            entries.append(
                HistoryEntry(
                    message_id=item.get("id", self._new_id()),
                    role=item["role"],
                    content=content_value or "",
                    timestamp=item.get("timestamp", self._current_timestamp()),
                    response_id=item.get("response_id"),
                    input=input_blocks,
                    output=sanitized_output,
                )
            )
        return entries

    def _write_entries(self, entries: List[HistoryEntry]) -> None:
        """Write the full entry list back to disk as YAML."""
        if not entries:
            if self._path.exists():
                self._path.unlink()
            return
        serializable = []
        for entry in entries:
            payload: dict[str, object] = {
                "id": entry.message_id,
                "role": entry.role,
                "timestamp": entry.timestamp,
            }
            if not entry.input and not entry.output and entry.content:
                payload["content"] = self._format_content(entry.content)
            if entry.response_id:
                payload["response_id"] = entry.response_id
            if entry.input:
                payload["input"] = self._prepare_for_dump(entry.input)
            if entry.output:
                payload["output"] = self._prepare_for_dump(entry.output)
            serializable.append(self._prepare_for_dump(payload))
        self._path.write_text(
            yaml.safe_dump(
                serializable,
                sort_keys=False,
                allow_unicode=True,
                width=80,
            ),
            encoding="utf-8",
        )

    def _build_entry(
        self,
        *,
        role: str,
        content: str,
        response_id: Optional[str] = None,
        input_blocks: Optional[List[Dict[str, object]]] = None,
        output: Optional[List[Dict[str, object]]] = None,
    ) -> HistoryEntry:
        """Create a new entry with generated metadata."""
        return HistoryEntry(
            message_id=self._new_id(),
            role=role,
            content=content,
            response_id=response_id,
            input=input_blocks,
            output=output,
            timestamp=self._current_timestamp(),
        )

    @staticmethod
    def _current_timestamp() -> str:
        """Return an ISO-8601 timestamp with UTC timezone."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_id() -> str:
        """Return a random UUID4 string for message identification."""
        return str(uuid4())

    @staticmethod
    def _format_content(content: str) -> str:
        """Format content for YAML output."""
        if "\n" in content:
            return LiteralString(content)
        return content

    @staticmethod
    def _prepare_for_dump(value: object) -> object:
        """Recursively wrap multiline strings so YAML emits literal block scalars."""
        if isinstance(value, str):
            if "\n" in value:
                stripped_lines = [line.rstrip() for line in value.splitlines()]
                normalized = "\n".join(stripped_lines)
                if value.endswith("\n"):
                    normalized = normalized + "\n"
                return LiteralString(normalized)
            return value
        if isinstance(value, list):
            return [ChatHistory._prepare_for_dump(item) for item in value]
        if isinstance(value, dict):
            return {key: ChatHistory._prepare_for_dump(val) for key, val in value.items()}
        return value

    def build_context(self) -> List[Dict[str, object]]:
        """Construct the Responses API context from stored entries."""
        context: List[Dict[str, object]] = []
        for entry in self._read_entries():
            if entry.role == "user":
                context.append(
                    {
                        "role": "user",
                        "content": self._convert_user_blocks(entry),
                    }
                )
            elif entry.role == "assistant":
                output_items = self._sanitize_output(entry.output)
                if output_items:
                    context.extend(output_items)
                elif entry.content:
                    context.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": entry.content}],
                        }
                    )
        return context

    def remove_last_turn(self) -> bool:
        """Remove the most recent user and assistant messages."""
        entries = self._read_entries()
        assistant_index = next(
            (idx for idx in range(len(entries) - 1, -1, -1) if entries[idx].role == "assistant"),
            None,
        )
        if assistant_index is None:
            return False
        user_index = next(
            (idx for idx in range(assistant_index - 1, -1, -1) if entries[idx].role == "user"),
            None,
        )
        if user_index is None:
            return False
        del entries[assistant_index]
        del entries[user_index]
        self._write_entries(entries)
        return True

    def _migrate_if_needed(self) -> None:
        """Convert legacy JSONL history files into the new YAML format."""
        legacy_path = self._path.with_suffix(".jsonl")
        if not legacy_path.exists():
            return
        current = self._path.read_text(encoding="utf-8").strip()
        if current:
            return
        entries: List[HistoryEntry] = []
        with legacy_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                input_blocks = payload.get("input")
                sanitized_output = self._sanitize_output(payload.get("output"))
                content_value = payload.get("content")
                if content_value is None:
                    if payload.get("role") == "user":
                        content_value = self._display_from_input(input_blocks)
                    elif payload.get("role") == "assistant":
                        content_value = self._display_from_output(sanitized_output)
                entries.append(
                    HistoryEntry(
                        message_id=payload.get("message_id", self._new_id()),
                        role=payload["role"],
                        content=content_value or "",
                        timestamp=payload.get("timestamp", self._current_timestamp()),
                        response_id=payload.get("response_id"),
                        input=input_blocks,
                        output=sanitized_output,
                    )
                )
        if entries:
            self._write_entries(entries)
        try:
            legacy_path.rename(legacy_path.with_suffix(".jsonl.bak"))
        except FileExistsError:
            legacy_path.unlink()

    @staticmethod
    def _sanitize_output(output: Optional[List[Dict[str, object]]]) -> List[Dict[str, object]]:
        """Return output items copied to plain dicts, dropping unset/None fields."""
        if not output:
            return []
        sanitized: List[Dict[str, object]] = []
        for item in output:
            sanitized.append(
                {
                    key: value
                    for key, value in item.items()
                    if value is not None
                }
            )
        return sanitized

    @staticmethod
    def _display_from_input(blocks: Optional[List[Dict[str, object]]]) -> str:
        """Reconstruct a human-readable view of user input blocks."""
        if not blocks:
            return ""
        parts: List[str] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "asset_image":
                source = block.get("source")
                if isinstance(source, str) and source:
                    parts.append(source)
                else:
                    path = block.get("path")
                    if path:
                        parts.append(f"[asset:{Path(str(path)).name}]")
            else:
                parts.append(str(block))
        return "".join(parts)

    @staticmethod
    def _display_from_output(blocks: List[Dict[str, object]]) -> str:
        """Construct assistant display text from response output blocks."""
        if not blocks:
            return ""
        for block in blocks:
            if block.get("type") != "message":
                continue
            contents = block.get("content", [])
            if not isinstance(contents, list):
                continue
            text_parts: List[str] = []
            for item in contents:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    text_parts.append(str(item.get("text", "")))
            if text_parts:
                return "".join(text_parts)
        return ""

    def _convert_user_blocks(self, entry: HistoryEntry) -> List[Dict[str, object]]:
        """Convert stored user blocks into API-ready content."""
        if entry.input:
            blocks: List[Dict[str, object]] = []
            for block in entry.input:
                if block.get("type") == "text":
                    blocks.append({"type": "input_text", "text": str(block.get("text", ""))})
                elif block.get("type") == "asset_image":
                    asset_path = Path(str(block.get("path")))
                    mime_type = str(block.get("mime_type", "image/png"))
                    blocks.append(
                        {
                            "type": "input_image",
                            "image_url": encode_asset(asset_path, mime_type),
                        }
                    )
                elif block.get("type") == "input_image":
                    image_ref = block.get("image_url")
                    if isinstance(image_ref, dict):
                        url = str(image_ref.get("url", ""))
                    else:
                        url = str(image_ref or "")
                    blocks.append({"type": "input_image", "image_url": url})
            if blocks:
                return blocks
        return [{"type": "input_text", "text": entry.content}]
