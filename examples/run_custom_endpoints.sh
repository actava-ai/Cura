#!/usr/bin/env bash
# The same harness against other OpenAI-compatible endpoints — pick the block you need.
set -euo pipefail

# --- Local vLLM ------------------------------------------------------------
# vllm serve Qwen/Qwen3-32B --port 8000
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"   # placeholder for keyless local servers
cura-eval medxpertqa --subset text --limit 50 \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-32B \
  --api-key-env VLLM_API_KEY

# --- Ollama ----------------------------------------------------------------
# ollama serve && ollama pull llama3.1
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-EMPTY}"
cura-eval healthbench --variant hard --limit 20 \
  --base-url http://localhost:11434/v1 \
  --model llama3.1 \
  --api-key-env OLLAMA_API_KEY

# --- OpenRouter ------------------------------------------------------------
# Pin reasoning-returning providers so reasoning_content is captured reliably.
cura-eval healthbench-professional --limit 25 \
  --base-url https://openrouter.ai/api/v1 \
  --model moonshotai/kimi-k2.6 \
  --api-key-env OPENROUTER_API_KEY \
  --extra-body '{"reasoning": {"enabled": true}}'

# --- OpenAI ----------------------------------------------------------------
# Reasoning models reject sampling params: omit temperature.
cura-eval agentclinic --dataset MedQA --limit 10 \
  --base-url https://api.openai.com/v1 \
  --model gpt-5.5 \
  --api-key-env OPENAI_API_KEY \
  --temperature none
