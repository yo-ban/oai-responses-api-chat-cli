# Purpose: Provide the command-line interface for the Responses API chat client.
# Why: A dedicated CLI module keeps entrypoint logic organized and testable.
"""CLI entrypoint implementing an interactive chat loop."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import re
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple
from uuid import UUID, uuid4

from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.document import Document  # type: ignore
from prompt_toolkit.history import InMemoryHistory  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore

try:  # noqa: SIM105
    import termios  # type: ignore
except ImportError:  # pragma: no cover
    termios = None  # type: ignore

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAIError

from .api_client import ResponsesChatClient
from .config import (
    DEFAULT_CONFIG_PATH,
    DATA_DIR,
    ensure_config,
    load_config,
    load_prompt,
    save_prompt,
)
from .assets import AssetStore, ConversationAssets, encode_asset
from .history import ChatHistory, HistoryEntry

DEFAULT_HISTORY_DIR = DATA_DIR / "history"
EXIT_COMMANDS = {":exit", ":quit"}
EXIT_TRIGGER = "__PROMPT_EXIT__"


def _prompt_continuation(width: int, line_number: int, wrap_count: int) -> str:
    """Render multiline continuation without leading ellipses."""
    if line_number == 0:
        return ""
    return " " * len("You> ")


def _create_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("enter", eager=True)
    def _(event) -> None:
        text = event.current_buffer.text
        if text.strip():
            event.app.exit(result=text)
        else:
            event.current_buffer.reset(Document(text="", cursor_position=0))

    @kb.add("c-c", eager=True)
    def _(event) -> None:
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset(Document(text="", cursor_position=0))
        else:
            event.app.exit(result=EXIT_TRIGGER)

    return kb


class InputManager:
    """Handle interactive user input via prompt_toolkit."""

    def __init__(self, session: Optional["PromptSession"] = None) -> None:
        if session is not None:
            self._session = session
            return
        self._session = PromptSession(
            multiline=True,
            history=InMemoryHistory(),
            key_bindings=_create_key_bindings(),
            prompt_continuation=_prompt_continuation,
        )

    def read(self) -> Tuple[str, bool]:
        """Return user input and whether exit was requested."""
        while True:
            try:
                result = self._session.prompt("You> ")
            except KeyboardInterrupt:
                buffer = self._session.default_buffer
                if buffer.text:
                    buffer.reset(Document(text="", cursor_position=0), append_to_history=False)
                    continue
                return "", True
            except EOFError:
                return "", True
            if result == EXIT_TRIGGER:
                return "", True
            if not result.strip():
                continue
            return result, False


def launch() -> None:
    """Entrypoint invoked by the console script."""
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "set-prompt":
        _handle_set_prompt(args)
        return
    try:
        _run_chat(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Conversation saved.", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive CLI guard.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="chatcli",
        description="CLI for OpenAI Responses API using GPT-5.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Start an interactive chat session.")
    _shared_connection_args(run_parser)
    run_parser.add_argument(
        "--conversation-id",
        type=str,
        default=None,
        help="Conversation UUID to resume. Omit to start a fresh session.",
    )
    run_parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Custom history file path. Overrides --conversation-id when provided.",
    )
    run_parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )

    prompt_parser = subparsers.add_parser(
        "set-prompt",
        help="Update the developer prompt file from provided text.",
    )
    prompt_parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    prompt_parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="New developer prompt text.",
    )
    return parser


def _shared_connection_args(parser: argparse.ArgumentParser) -> None:
    """Attach connection arguments shared by multiple subcommands."""
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.openai.com/v1/",
        help="OpenAI base URL (defaults to OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model identifier from configuration.",
    )


def _handle_set_prompt(args: argparse.Namespace) -> None:
    """Persist a new developer prompt based on CLI arguments."""
    config_path: Path = args.config_file
    save_prompt(args.text, config_path)
    print(f"Developer prompt saved to {config_path}")


def _run_chat(args: argparse.Namespace) -> None:
    """Execute the interactive chat loop."""
    config_path: Path = args.config_file
    config_created = ensure_config(config_path)
    if config_created:
        print(f"Created default config at {config_path}")
    config = load_config(config_path, ensure=False)
    base_url_raw = _resolve_setting(
        args.base_url,
        config.get("base_url"),
        "OPENAI_BASE_URL",
        "base URL",
    )
    base_url = _normalize_base_url(base_url_raw)
    api_key = _resolve_setting(
        args.api_key,
        config.get("api_key"),
        "OPENAI_API_KEY",
        "API key",
    )
    model = args.model or config.get("model") or "gpt-5"
    max_output_tokens = config.get("max_output_tokens")
    reasoning_config = {
        key: value
        for key, value in (config.get("reasoning") or {}).items()
        if value is not None
    }

    conversation_id = _ensure_conversation_id(args.conversation_id)
    history_path = _determine_history_path(args.history_file, conversation_id)

    developer_prompt = config["developer_prompt"]
    history = ChatHistory(history_path)
    context = history.build_context()
    asset_store = AssetStore().for_conversation(conversation_id)

    client = ResponsesChatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    _print_intro(config_path, history_path, conversation_id)
    _display_history(history.entries())

    input_manager = InputManager()

    while True:
        user_input, exit_requested = input_manager.read()
        if exit_requested:
            _print_farewell(history, conversation_id)
            break
        if user_input in EXIT_COMMANDS:
            _print_farewell(history, conversation_id)
            break
        if user_input == ":help":
            print(_render_help())
            continue
        if user_input == ":showprompt":
            print(load_prompt(config_path))
            continue
        if user_input == ":reset":
            history.clear()
            context = []
            print("History cleared.")
            continue
        if user_input == ":undo":
            if history.remove_last_turn():
                context = history.build_context()
                print("Last turn removed.")
                _display_history(history.entries())
            else:
                print("No turn available to remove.")
            continue

        developer_prompt = load_prompt(config_path)
        try:
            display_content, user_blocks = _prepare_user_blocks(user_input, asset_store)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Assistant> {exc}")
            continue
        api_blocks = _blocks_to_api(user_blocks)
        context.append({"role": "user", "content": api_blocks})
        status = "... generating ..."
        _print_status(status)
        try:
            with _suspend_input():
                result = client.send_message(
                    context=context,
                    developer_prompt=developer_prompt,
                    max_output_tokens=max_output_tokens,
                    reasoning=reasoning_config or None,
                )
        except KeyboardInterrupt:
            _clear_status(status)
            context.pop()
            print("\nGeneration cancelled.")
            continue
        except APIConnectionError as exc:
            _clear_status(status)
            context.pop()
            print(f"Assistant> 接続エラー: {_format_error(exc)}", file=sys.stderr)
            continue
        except OpenAIError as exc:
            _clear_status(status)
            context.pop()
            print(f"Assistant> APIエラー: {_format_error(exc)}", file=sys.stderr)
            continue
        _clear_status(status)
        history.record_turn(
            user_content=display_content,
            assistant_content=result.message_text,
            response_id=result.response_id,
            input_blocks=user_blocks,
            output=result.output_items,
        )
        context = history.build_context()
        print(f"Assistant> {result.message_text}")

def _print_farewell(history: ChatHistory, conversation_id: str) -> None:
    """Display exit message with resume hint when history exists."""
    print("Goodbye.")
    if history.path.exists():
        print(f"Resume with: chatcli run --conversation-id {conversation_id}")


def _render_help() -> str:
    """Return a help string for inline command discovery."""
    return (
        "Keyboard shortcuts:\n"
        "  Enter            Send the current message.\n"
        "  Ctrl+J / Alt+Enter  Insert a newline without sending.\n"
        "  Ctrl+C           Clear the draft, or exit when already empty.\n"
        "\n"
        "Commands:\n"
        "  :help        Show this message.\n"
        "  :showprompt  Display the active developer prompt.\n"
        "  :reset       Clear stored history for a fresh session.\n"
        "  :undo        Remove the most recent turn.\n"
        "  :exit/:quit  End the session.\n"
    )


def _print_intro(config_path: Path, history_path: Path, conversation_id: str) -> None:
    """Display startup guidance for the operator."""
    print("Responses API Chat CLI (GPT-5)")
    print(f"Conversation id: {conversation_id}")
    print(f"Config file: {config_path}")
    print(f"History file: {history_path}")
    print("Type messages to send them to the model. Use :help for commands.")


def _resolve_setting(
    cli_value: Optional[str],
    config_value: Optional[str],
    env_var: str,
    label: str,
) -> str:
    """Resolve a configuration value prioritizing CLI args, config, then environment."""
    if cli_value:
        return cli_value
    if config_value:
        return config_value
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
    raise ValueError(f"Missing {label}. Provide via CLI, config, or set {env_var}.")


def _normalize_base_url(value: str) -> str:
    """Ensure the base URL includes the Responses API prefix when omitted."""
    if "openai.com" in value.lower():
        return value
    trimmed = value.rstrip("/")
    lower_trimmed = trimmed.lower()
    if "/openai" not in lower_trimmed:
        return f"{trimmed}/openai/v1"
    return trimmed


def _format_error(exc: Exception) -> str:
    """Return a human-readable error string including root cause information."""
    message = str(exc) or exc.__class__.__name__
    cause = getattr(exc, "__cause__", None)
    if cause:
        return f"{message} (原因: {cause})"
    return message


def _ensure_conversation_id(value: Optional[str]) -> str:
    """Return a UUID string, generating one when omitted."""
    if not value:
        return str(uuid4())
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError("conversation-id must be a valid UUID") from exc


def _determine_history_path(provided: Optional[Path], conversation_id: str) -> Path:
    """Derive the history file path from CLI arguments."""
    if provided:
        return provided
    DEFAULT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"chat_history_{conversation_id}.yaml"
    return DEFAULT_HISTORY_DIR / filename


def _display_history(entries: List[HistoryEntry]) -> None:
    """Print the stored conversation history."""
    if not entries:
        return
    print("Conversation history:")
    for entry in entries:
        if entry.role == "user":
            print(f"You> {entry.content}")
        elif entry.role == "assistant":
            print(f"Assistant> {_assistant_text(entry)}")


def _assistant_text(entry: HistoryEntry) -> str:
    """Extract assistant text from stored output if available."""
    for block in entry.output or []:
        if block.get("type") == "message":
            contents = block.get("content", [])
            if isinstance(contents, list):
                for item in contents:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        text = item.get("text")
                        if isinstance(text, str):
                            return text
    return entry.content


def _prepare_user_blocks(raw_input: str, asset_store: ConversationAssets) -> tuple[str, List[Dict[str, object]]]:
    """Return display content and stored blocks for a user message."""
    pattern = re.compile(r"@(\S+)")
    blocks: List[Dict[str, object]] = []
    cursor = 0
    has_match = False

    for match in pattern.finditer(raw_input):
        has_match = True
        preceding = raw_input[cursor:match.start()]
        if preceding:
            blocks.append({"type": "text", "text": preceding})
        asset_token = match.group(1)
        if not asset_token:
            raise ValueError("Specify a path after '@'.")
        path = Path(asset_token).expanduser()
        stored_assets = asset_store.store(path)
        for index, asset in enumerate(stored_assets, start=1):
            source_label = f"@{asset_token}"
            if len(stored_assets) > 1:
                source_label = f"{source_label}#{index}"
            blocks.append(
                {
                    "type": "asset_image",
                    "path": str(asset.path),
                    "mime_type": asset.mime_type,
                    "source": source_label,
                }
            )
        cursor = match.end()

    trailing = raw_input[cursor:]
    if trailing:
        blocks.append({"type": "text", "text": trailing})

    if has_match:
        if not blocks:
            raise ValueError("Attachment references must include accompanying text or valid files.")
        return raw_input, blocks

    return raw_input, [{"type": "text", "text": raw_input}]


def _blocks_to_api(blocks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Convert stored blocks into Responses API content items."""
    api_blocks: List[Dict[str, object]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            api_blocks.append({"type": "input_text", "text": str(block.get("text", ""))})
        elif block_type == "asset_image":
            path_value = Path(str(block.get("path")))
            mime_value = str(block.get("mime_type", "image/png"))
            data_url = encode_asset(path_value, mime_value)
            api_blocks.append({"type": "input_image", "image_url": data_url})
    return api_blocks


def _print_status(message: str) -> None:
    """Render a transient status message on the current line."""
    if message:
        print(message, flush=True)


def _clear_status(message: str) -> None:
    """Clear a previously rendered status message."""
    if not message:
        return
    sys.stdout.write("\033[F\033[K")
    sys.stdout.flush()


@contextmanager
def _suspend_input() -> Iterator[None]:
    """Disable input echo while awaiting a response."""
    if termios is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    new_attrs = termios.tcgetattr(fd)
    new_attrs[3] &= ~(termios.ECHO | termios.ICANON)
    termios.tcsetattr(fd, termios.TCSANOW, new_attrs)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
