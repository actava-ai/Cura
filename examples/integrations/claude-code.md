# Claude Code × Cura

Claude Code talks the Anthropic API, so it uses the Cura gateway's **Anthropic-compatible
endpoint** — note the base URL is `/anthropic`, not `/v1`.

```bash
# Install Claude Code (once)
npm install -g @anthropic-ai/claude-code

# Point it at the Cura gateway's Anthropic-compatible endpoint, then start:
export ANTHROPIC_BASE_URL=https://inference.actava.ai/anthropic
export ANTHROPIC_AUTH_TOKEN=$ACTAVA_API_KEY
export ANTHROPIC_MODEL=actava/cura-soar
claude
```
