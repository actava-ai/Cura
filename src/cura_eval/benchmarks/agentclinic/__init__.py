"""AgentClinic: simulated multi-turn clinical consultations (MedQA / NEJM cases).

Port of AgentClinic (Schmidgall et al., https://github.com/SamuelSchmidgall/AgentClinic, MIT).
The doctor — the model under test — interviews a simulated patient, orders tests from a
measurement agent, and must commit to a diagnosis within a fixed question budget; a
moderator grades the diagnosis against gold. Patient / measurement / moderator NPCs run on
the judge endpoint; the doctor runs on any OpenAI-compatible endpoint.
"""

from cura_eval.benchmarks.agentclinic.runner import run_agentclinic

__all__ = ["run_agentclinic"]
