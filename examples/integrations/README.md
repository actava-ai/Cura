# Use Cura in local agents

The Cura API speaks the OpenAI chat-completions dialect at
`https://inference.actava.ai/v1` only (bearer auth with `ACTAVA_API_KEY`, model id
`actava/cura-soar`). That means most local agents can use Cura by adding it as a custom /
OpenAI-compatible provider; Claude Code, which speaks Anthropic's Messages protocol, goes
through a local translation proxy instead (see [`claude-code.md`](claude-code.md)).

These configs mirror the official integration docs at
[actava.ai/cura/docs](https://actava.ai/cura/docs):

| Agent | Config |
|---|---|
| Claude Code | [`claude-code.md`](claude-code.md) |
| Hermes Agent | [`hermes.md`](hermes.md) |
| OpenClaw | [`openclaw/openclaw.json`](openclaw/openclaw.json) |
| Cline (VS Code) | [`cline.md`](cline.md) |
| Roo Code (VS Code) | [`roo-code.md`](roo-code.md) |

Any other tool with an "OpenAI-compatible" provider option works the same way:

```
Base URL:   https://inference.actava.ai/v1
API key:    $ACTAVA_API_KEY
Model:      actava/cura-soar
Context:    256000    Vision: yes
```

And plain SDK use is a two-line migration from OpenAI:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ACTAVA_API_KEY"],
    base_url="https://inference.actava.ai/v1",
)
response = client.chat.completions.create(
    model="actava/cura-soar",
    messages=[{"role": "user", "content": "Summarize the escalation criteria for chest pain triage."}],
)
print(response.choices[0].message.content)
```
