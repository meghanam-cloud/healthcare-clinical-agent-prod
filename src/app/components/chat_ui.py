"""
Small render helpers kept separate from app.py so the entrypoint stays thin.
"""

import streamlit as st


def render_history():
    for turn in st.session_state.messages:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sql_query"):
                with st.expander("SQL used"):
                    st.code(turn["sql_query"], language="sql")


def append_message(role: str, content: str, sql_query: str = None):
    entry = {"role": role, "content": content}
    if sql_query:
        entry["sql_query"] = sql_query
    st.session_state.messages.append(entry)
