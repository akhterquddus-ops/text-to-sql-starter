"""Streamlit front end.

    streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from sqlassist.config import Settings, make_provider
from sqlassist.pipeline import Assistant
from sqlassist.semantic import SemanticLayer

st.set_page_config(page_title="Text-to-SQL Starter", page_icon="🗄️", layout="wide")


@st.cache_resource
def get_assistant() -> Assistant:
    s = Settings.from_env()
    return Assistant(
        make_provider(s),
        SemanticLayer.load(s.semantic_layer),
        s.db_path,
        max_attempts=s.max_attempts,
        row_limit=s.row_limit,
        query_timeout_s=s.query_timeout_s,
    )


def chart(df: pd.DataFrame) -> alt.Chart | None:
    """Draw a simple chart when the result has one label and one number."""
    if df is None or len(df) < 2 or df.shape[1] < 2:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric or not labels:
        return None
    x, y = labels[0], numeric[-1]
    is_time = x.endswith("month") or x.endswith("date")
    base = alt.Chart(df).encode(
        x=alt.X(x, sort=None if is_time else "-y", title=x.replace("_", " ")),
        y=alt.Y(y, title=y.replace("_", " ")),
        tooltip=list(df.columns),
    )
    return (base.mark_line(point=True) if is_time else base.mark_bar()).properties(height=320)


st.title("🗄️ Text-to-SQL Starter")
st.caption("Ask a question about the Demo Retail Co. data in plain English. "
           "You get the answer, a chart and the exact SQL that produced it. All data is synthetic.")

try:
    assistant = get_assistant()
except Exception as e:  # missing DB or misconfigured provider
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("Try asking")
    samples = [
        "What was net revenue by region in 2025?",
        "Show monthly net revenue for 2025",
        "Top 10 customers by lifetime revenue",
        "Gross margin percent by category",
        "Cancellation rate by channel",
        "Delete all cancelled orders",
    ]
    for q in samples:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q
    st.divider()
    st.caption(f"Model: `{assistant.provider.name}`")
    st.caption(f"Approved views: {', '.join(sorted(assistant.semantic.allowed_tables))}")

question = st.text_input("Your question", key="question",
                         placeholder="e.g. Which product category grew most in 2025?")

if question:
    with st.spinner("Thinking..."):
        ans = assistant.ask(question)

    if not ans.ok:
        st.warning(f"I couldn't answer that safely. {ans.error or ''}")
    else:
        st.success(ans.summary)
        c = chart(ans.data)
        if c is not None:
            st.altair_chart(c, use_container_width=True)
        st.dataframe(ans.data, use_container_width=True, hide_index=True)
        with st.expander("SQL that was run"):
            st.code(ans.sql, language="sql")

    with st.expander(f"How this answer was made ({len(ans.attempts)} attempt(s), {ans.seconds}s)"):
        for i, a in enumerate(ans.attempts, 1):
            st.markdown(f"**Attempt {i}:** {'✅ ran successfully' if a.error is None else '❌ ' + a.error}")
            st.code(a.raw_sql or "(no SQL)", language="sql")
