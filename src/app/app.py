"""
Databricks App entrypoint - Streamlit multi-turn SQL agent chatbot.
Run locally with: streamlit run src/app/app.py
"""

import uuid
import streamlit as st

from src.agent.graph import run_agent
from src.memory.short_term import log_turn, get_recent_turns
from src.app.components.chat_ui import render_history, append_message
from src.observability.langfuse_tracer import get_tracer, trace_span

st.set_page_config(page_title="Clinical Care Assistant", page_icon="🩺")
st.title("🩺 Clinical Care Coordination Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "anonymous"
if "messages" not in st.session_state:
    # Initialise tracing gracefully – a Langfuse connectivity/credential
    # failure must NOT prevent the chat UI from rendering.
    try:
        _tracer_init = get_tracer()
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Langfuse tracer unavailable; continuing without tracing.",
            exc_info=True,
        )
        from src.observability.langfuse_tracer import _NoOpTracer
        _tracer_init = _NoOpTracer()

    _init_trace = _tracer_init.trace(
        name="session_init_memory_load",
        user_id=st.session_state.user_id,
        session_id=st.session_state.session_id,
    )
    with trace_span(_init_trace, "memory_load_history", input=st.session_state.session_id) as _mem_span:
        persisted = get_recent_turns(st.session_state.session_id)
        _mem_span.update(output={"turns_loaded": len(persisted)})
    _init_trace.update(output={"turns_loaded": len(persisted)})
    st.session_state.messages = [{"role": t["role"], "content": t["content"], "sql_query": t.get("sql_query")} for t in persisted]

render_history()

question = st.chat_input("Ask about patients, risk tiers, or clinical guidelines...")
if question:
    append_message("user", question)
    with st.chat_message("user"):
        st.markdown(question)

    # Create the top-level Langfuse trace for this chat turn so both memory
    # operations and the agent graph are visible under a single trace.
    try:
        _tracer = get_tracer()
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Langfuse tracer unavailable; continuing without tracing.",
            exc_info=True,
        )
        from src.observability.langfuse_tracer import _NoOpTracer
        _tracer = _NoOpTracer()
    _turn_trace = _tracer.trace(
        name="clinical_agent_turn",
        user_id=st.session_state.user_id,
        session_id=st.session_state.session_id,
        input=question,
    )

    with trace_span(_turn_trace, "memory_log_user_turn", input=question) as _mem_span:
        log_turn(st.session_state.session_id, "user", question)
        _mem_span.update(output="logged")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Re-read short-term memory for every turn so follow-up questions
            # are planned from the persisted Delta table, not only Streamlit state.
            persisted_turns = get_recent_turns(st.session_state.session_id)
            if persisted_turns:
                # The current user turn was logged immediately above; exclude it
                # from prior conversation context passed to the agent.
                history_for_agent = persisted_turns[:-1] if (
                    persisted_turns[-1].get("role") == "user"
                    and persisted_turns[-1].get("content") == question
                ) else persisted_turns
            else:
                history_for_agent = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "sql_query": m.get("sql_query"),
                    }
                    for m in st.session_state.messages[:-1]
                ]

            result = run_agent(
                question=question,
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_id,
                chat_history=history_for_agent,
                trace=_turn_trace,
            )
            answer = result.get("final_response", "Sorry, something went wrong.")
            sql_used = result.get("sql_query")
            st.markdown(answer)
            if sql_used and not result.get("blocked"):
                with st.expander("SQL used"):
                    st.code(sql_used, language="sql")

    append_message("assistant", answer, sql_query=sql_used)

    with trace_span(_turn_trace, "memory_log_assistant_turn", input=answer[:200]) as _mem_span:
        log_turn(st.session_state.session_id, "assistant", answer, sql_query=sql_used)
        _mem_span.update(output="logged", metadata={"sql_query": sql_used})

    _turn_trace.update(output=answer)
