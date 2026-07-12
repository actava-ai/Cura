#!/usr/bin/env bash
# Run the full medical benchmark suite against the Cura API.
#
# Requirements:
#   export ACTAVA_API_KEY=...   # candidate (https://actava.ai/cura)
#   export OPENAI_API_KEY=...   # rubric judges + AgentClinic NPCs
#   docker run -d -p 8080:8080 jyxsu6/medagentbench:latest   # for medagentbench
#
# Drop the --limit flags for full runs.
set -euo pipefail

LIMIT="${LIMIT:-25}"

cura-eval healthbench --variant hard --limit "$LIMIT"
cura-eval healthbench-professional --limit "$LIMIT"
cura-eval medxpertqa --subset text --limit "$LIMIT"
cura-eval medxpertqa --subset mm --limit "$LIMIT"
cura-eval agentclinic --dataset MedQA --limit "$LIMIT"
cura-eval agentclinic --dataset NEJM --limit "$LIMIT"
cura-eval medagentbench --limit "$LIMIT"
