"""cura-eval: model-agnostic medical evaluation harness.

Every benchmark drives the model under test through one seam — an OpenAI-compatible
chat-completions endpoint (`cura_eval.client.ChatClient`) — so the same harness evaluates
the Cura API, a local vLLM/Ollama/SGLang server, OpenRouter, or any other compatible server.
"""

__version__ = "0.1.0"

from cura_eval.client import ChatClient, ChatResult

__all__ = ["ChatClient", "ChatResult", "__version__"]
