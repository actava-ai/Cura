# Claude Code × Cura

Claude Code speaks Anthropic's Messages protocol, but the Cura gateway exposes the OpenAI
Chat Completions API at `/v1` only — so Claude Code connects through a small local proxy,
[CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI), which translates between the two.
The proxy runs on your machine; your `ACTAVA_API_KEY` stays in its `config.yaml` and Claude Code
itself only needs a dummy token.

## 1. Install CLIProxyAPI

```bash
# Linux / macOS installer
curl -fsSL https://raw.githubusercontent.com/router-for-me/cliproxyapi-installer/refs/heads/master/cliproxyapi-installer | bash
```

Or run it with Docker (listens on port 8317):

```bash
docker run --rm -p 8317:8317 \
  -v "$PWD/config.yaml":/CLIProxyAPI/config.yaml \
  eceasy/cli-proxy-api:latest
```

## 2. Connect it to Cura

Register Cura as an OpenAI-compatible provider in the proxy's `config.yaml`:

```yaml
port: 8317

openai-compatibility:
  - name: "cura"
    base-url: "https://inference.actava.ai/v1"
    api-key-entries:
      - api-key: "YOUR_ACTAVA_API_KEY"
    models:
      - name: "actava/cura-soar"
        alias: "cura-soar"
```

## 3. Point Claude Code at the proxy

Add an alias to `~/.zshrc` / `~/.bashrc` that routes Claude Code through the proxy and pins
both the main agent and subagents to Cura:

```bash
alias claudex='ANTHROPIC_BASE_URL=http://127.0.0.1:8317 \
ANTHROPIC_AUTH_TOKEN=sk-dummy \
CLAUDE_CODE_SUBAGENT_MODEL=cura-soar \
CLAUDE_CODE_ALWAYS_ENABLE_EFFORT=1 \
CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=3 \
ENABLE_TOOL_SEARCH=false \
claude --model cura-soar'
```

Then start a session with `claudex`.

Notes:

- `ANTHROPIC_AUTH_TOKEN=sk-dummy` is intentional — the real key lives in the proxy's
  `config.yaml`, and translation happens locally.
- `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=3` and `ENABLE_TOOL_SEARCH=false` keep request
  patterns conservative for a non-Anthropic backend.
- Agentic sessions make many model calls per task — keep an eye on usage during long runs.
