# Responses API Chat CLI

This lightweight command-line client makes it easy to chat with the OpenAI Responses API (and compatible Azure endpoints) directly from your terminal. It was born out of the need to iterate on prompts and LLM features using real API calls. The tool keeps prior turns, feeds reasoning tokens back into each request so you get the full model experience, and supports encrypted reasoning items for zero-data-retention workflows. Use it whenever you want to tweak prompts quickly without spinning up a GUI playground.

## Requirements

- Python 3.11+
- Configured OpenAI (or Azure OpenAI) resource with a GPT-5 deployment
- Connection settings via `config/settings.yaml` (preferred) or environment variables:
- `base_url` or `OPENAI_BASE_URL`
- `api_key` or `OPENAI_API_KEY`

The CLI defaults to `--model gpt-5`. Override it via CLI flag when you need a different deployment. Leave `base_url` empty when calling the public OpenAI API; the SDK default (`https://api.openai.com/v1`) is used automatically. Provide the Azure resource root (for example `https://<resource>.openai.azure.com`) when targeting Azure and the CLI will append `/openai/v1` when missing. Settings defined in `config/settings.yaml` take precedence; the CLI falls back to environment variables (including those loaded from a local `.env` file) when a value is not present in the config.

You can define multiple profiles under `profiles` in `config/settings.yaml` and select them with `--profile <name>`.

Example `config/settings.yaml` structure:

```yaml
developer_prompt: |
  You are an assistant that provides concise, accurate responses while citing assumptions.
  Keep answers in Japanese unless the user explicitly requests another language.
base_url: null
api_key: null
model: gpt-5
max_output_tokens: null
reasoning:
  effort: medium
  summary: null
pdf_render_dpi: 200
profiles:
  "1":
    base_url: https://api.openai.com/v1
    api_key: sk-xxx
    model: gpt-5
    max_output_tokens: null
    reasoning:
      effort: medium
      summary: null
  "2":
    base_url: https://my-azure-resource.openai.azure.com
    api_key: azure-key
    model: gpt-5-azure
    max_output_tokens: 2048
    reasoning:
      effort: high
      summary: null
```

## Installation

Choose one of the following setups.

### Using `uv` (recommended)

```bash
uv sync
source .venv/bin/activate
```

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Configuration and chat history are stored under `~/.responses-cli/` (`config/settings.yaml`, `data/history`, and `data/assets`). Override paths with `--config-file` or `--history-file` if you need a different location.

## Usage

Update the developer prompt (optional):

```bash
chatcli set-prompt --text "You are a helpful assistant."
```

Start an interactive session (a new UUID is generated if omitted):

```bash
chatcli run
```

Multi-line input is supported. Press Enter to send; insert a newline with `Ctrl+J` (or `Alt+Enter`). `Ctrl+C` clears the current draft, and when no text is present it exits the session. Empty leading lines are ignored so accidental blank submissions are dropped.

Send an image (or PDF) by referencing it with `@relative/path`:

```bash
You> @images/sample.png
```

The CLI copies image assets into `data/assets/<conversation>`; PDFs are converted to per-page PNGs automatically.

Resume a prior conversation by supplying its UUID:

```bash
chatcli run --resume 12345678-1234-5678-1234-567812345678
```

Use `chatcli run --resume` (without an id) to pick from the most recent histories.

Inline commands:

- `:help` – show available commands
- `:showprompt` – print the active developer prompt
- `:reset` – clear stored conversation history
- `:undo` – remove the most recent user/assistant turn
- `:exit` / `:quit` – end the session

Previously saved conversations are shown automatically when you resume a session so you can pick up where you left off. Chat histories are saved as YAML files named `data/chat_history_<conversation-uuid>.yaml`. Runtime settings (developer prompt, model, `max_output_tokens`, reasoning effort/summary) live in `config/settings.yaml`; edit this file to customise defaults. Each assistant turn stores the raw Responses API output (including encrypted reasoning items) so the Responses API can continue statelessly from edited histories.
