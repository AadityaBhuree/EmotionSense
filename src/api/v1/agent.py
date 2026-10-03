"""Clinical Reasoning Agent and Copilot Router."""

from fastapi import APIRouter
from src.agent import ClinicalReasoningAgent, ClinicalChatCopilot, ChatCopilotQuery, LLMProviderConfig
from src.api.deps import db
from src.api.v1.schemas import AgentSynthesizeRequest, AgentChatRequest

router = APIRouter(tags=["Agentic Reasoning"])


@router.get("/api/agent/providers")
async def get_agent_providers():
    """Returns supported LLM/SLM reasoning providers and active default."""
    return {
        "status": "success",
        "providers": ["rule_based", "ollama", "openai", "gemini"],
        "active_default": "rule_based",
    }


@router.post("/api/agent/synthesize")
async def synthesize_session_diagnostics(req: AgentSynthesizeRequest):
    """Executes Chain-of-Thought clinical diagnostic synthesis across multimodal session telemetry."""
    session_data = {
        "session_id": req.session_id,
        "candidate_id": req.subject_id,
        "assessment_type": req.assessment_type,
        "timeline_samples": req.timeline_samples or [],
        "anomalies": req.anomalies or [],
    }
    config = LLMProviderConfig(provider_name=req.provider_name or "rule_based")
    agent = ClinicalReasoningAgent(provider_config=config)
    synthesis = agent.synthesize_session(session_data)
    return {"status": "success", "synthesis": synthesis.model_dump()}


@router.post("/api/agent/chat")
async def chat_clinical_copilot(req: AgentChatRequest):
    """Answers clinician queries grounded in multimodal session telemetry."""
    config = LLMProviderConfig(provider_name=req.provider_name or "rule_based")
    copilot = ClinicalChatCopilot(provider_config=config)
    query = ChatCopilotQuery(
        session_id=req.session_id,
        question=req.question,
    )
    ctx = {"session_id": req.session_id}
    stored = db.get_session(req.session_id)
    if stored:
        ctx["timeline_samples"] = stored.get("timeline_samples", [])
        ctx["anomalies"] = stored.get("anomalies", [])
        ctx["candidate_id"] = stored.get("candidate_id", "Anonymous")
        ctx["assessment_type"] = stored.get("assessment_type", "Assessment")

    resp = copilot.query(query, session_context=ctx)
    return {"status": "success", "response": resp.model_dump()}
