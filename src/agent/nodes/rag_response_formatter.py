"""
Node: the RAG worker's generation step. Given the retrieved, reranked
clinical guideline context, produce a grounded draft answer citing source
pages. This draft is checked by the review agent before it is shown.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a grounded clinical guidelines assistant. Answer the user's question "
    "using ONLY the supplied context from the clinical practice guidelines manual. "
    "Rules:\n"
    "1. Do not invent facts that are not present in the context.\n"
    "2. If the context is insufficient, say the guidelines manual does not contain "
    "enough information rather than guessing.\n"
    "3. Mention the source page number(s) you drew on.\n"
    "4. Present this as guideline/protocol information, not as a diagnosis or "
    "personal medical advice for any specific patient."
)


def rag_response_formatter_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    context = state.get("rag_context", "")

    if not context.strip():
        draft = (
            "I couldn't find anything relevant in the clinical guidelines manual "
            "for that question. Could you rephrase it or narrow it down?"
        )
        with trace_span(trace, "rag_response_formatting", input={"empty_context": True}) as span:
            span.update(output=draft, metadata={"type": "empty_context_fallback"})
        return {**state, "draft_response": draft}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['question']}\n\nRetrieved context:\n{context}"),
    ]

    with trace_span(
        trace,
        "rag_response_formatting",
        input={"question": state["question"]},
        metadata={"context_length": len(context)},
    ) as span:
        gen = span.generation(
            name="llm_rag_generation",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={"temperature": settings.llm.temperature},
        )

        llm = get_llm()
        response = llm.invoke(messages)

        gen.update(output=response.content)
        gen.end()
        span.update(output=response.content[:1000])

    return {**state, "draft_response": response.content.strip()}
