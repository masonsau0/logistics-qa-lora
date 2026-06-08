"""Streamlit UI for the logistics QA model.

Talks to the FastAPI inference server (server/app.py). Two modes:
  - Generate: synchronous request to /generate.
  - Stream:   server-sent events from /generate/stream, token-by-token.

Designed for the demo loop: pick a sample question, watch the model answer in
real time, compare to the canned reference if you flip the toggle. Recruiters
should see what the model says AND that it's the fine-tuned adapter answering
(the badge top-right tells them).

Run:
    streamlit run ui/app.py

Assumes the server is on http://localhost:8000. Set SERVER_URL to point elsewhere
(e.g. a deployed Cloud Run / HF Spaces URL).
"""

from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")


st.set_page_config(page_title="Logistics QA - LoRA Qwen 2.5", page_icon="🚚", layout="wide")
st.title("Logistics QA - LoRA fine-tuned Qwen 2.5")
st.markdown(
    "**A 7B model QLoRA-fine-tuned on a 12K-example synthetic Q&A dataset covering "
    "freight, claims, routing, SOPs, and FMCSA / customs compliance.**"
)
st.caption(
    "Ask a domain question on the left. The fine-tuned model answers via the FastAPI "
    "server (`/generate` or `/generate/stream`)."
)


# ---------- Sidebar: examples + settings ----------

EXAMPLES = [
    "What's the deadline for filing a concealed-damage freight claim under U.S. law?",
    "What is the DIM factor for LTL shipments and why does it matter?",
    "Explain Carmack Amendment carrier liability in one paragraph.",
    "What's the difference between C-TPAT and AEO?",
    "How is freight class determined under NMFC?",
]

with st.sidebar:
    st.header("Examples")
    for ex in EXAMPLES:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    st.header("Generation settings")
    stream = st.toggle("Stream tokens", value=True, help="Use /generate/stream (SSE).")
    max_new_tokens = st.slider("max_new_tokens", 32, 1024, 384, step=32)
    temperature = st.slider("temperature", 0.0, 1.5, 0.2, step=0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, step=0.05)

    st.divider()
    st.caption(f"Server: {SERVER_URL}")


# ---------- Health check + adapter badge ----------

col_left, col_right = st.columns([4, 1])
try:
    health = httpx.get(f"{SERVER_URL}/health", timeout=3.0)
    if health.status_code == 200:
        body = health.json()
        backend = body.get("backend", "?")
        with col_left:
            st.success(f"✓ Server reachable - backend: {backend}")
        with col_right:
            badge = "🟢 fine-tuned" if backend == "model" else "🟡 mock"
            st.markdown(
                f"<div style='text-align:right;font-size:0.9rem'>{badge}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.error(f"Server returned {health.status_code}")
        st.stop()
except httpx.HTTPError:
    st.error(
        f"Can't reach server at {SERVER_URL}. Start it with: uvicorn server.app:app --port 8000"
    )
    st.stop()


# ---------- Question input ----------

default_q = st.session_state.pop("prefill", "")
question = st.text_area(
    "Question",
    value=default_q,
    height=80,
    placeholder="What is dimensional weight and why does it matter for parcel pricing?",
)

submit = st.button("Ask", type="primary")


# ---------- Generate / stream ----------


def _payload() -> dict:
    return {
        "question": question,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _stream_answer() -> tuple[str, float]:
    """Read the SSE stream and yield text into a Streamlit placeholder.
    Returns (full_text, elapsed_seconds)."""
    placeholder = st.empty()
    parts: list[str] = []
    t0 = time.time()
    with httpx.stream("POST", f"{SERVER_URL}/generate/stream", json=_payload(), timeout=180.0) as r:
        if r.status_code != 200:
            placeholder.error(f"Stream failed: {r.status_code}")
            return "", time.time() - t0
        for raw in r.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[len("data:") :].strip()
            if chunk == "[DONE]":
                break
            if chunk.startswith("[ERROR]"):
                placeholder.error(chunk)
                break
            parts.append(chunk)
            placeholder.markdown("".join(parts))
    return "".join(parts), time.time() - t0


def _generate_answer() -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = httpx.post(f"{SERVER_URL}/generate", json=_payload(), timeout=180.0)
    except httpx.HTTPError as e:
        st.error(f"Request failed: {e}")
        return "", time.time() - t0
    elapsed = time.time() - t0
    if resp.status_code != 200:
        st.error(f"Server returned {resp.status_code}: {resp.text}")
        return "", elapsed
    return resp.json().get("answer", ""), elapsed


if submit and question.strip():
    st.subheader("Answer")
    if stream:
        with st.spinner("Streaming..."):
            answer, elapsed = _stream_answer()
    else:
        with st.spinner("Generating..."):
            answer, elapsed = _generate_answer()
            if answer:
                st.markdown(answer)

    if answer:
        a, b = st.columns(2)
        a.metric("Tokens (approx)", len(answer.split()))
        b.metric("Elapsed", f"{elapsed:.1f}s")

        with st.expander("Raw payload"):
            st.code(json.dumps(_payload(), indent=2), language="json")
