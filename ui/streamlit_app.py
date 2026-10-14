"""Streamlit client for the BankGuard RAG FastAPI service."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("BANKGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="BankGuard RAG", page_icon="🏦", layout="centered")
st.title("🏦 BankGuard RAG")
st.caption("Ask about banking and KYC policy. Answers are grounded in the displayed sources.")

with st.form("query_form"):
    question = st.text_area(
        "Your question",
        placeholder="How often must KYC be updated for high-risk customers?",
        max_chars=2000,
        height=110,
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Checking the available policy sources…"):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": question.strip()},
                    timeout=(5, 180),
                )
            if response.ok:
                result = response.json()
                if result.get("refused"):
                    st.warning(result["answer"])
                else:
                    st.markdown(result["answer"])

                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for i, source in enumerate(sources, start=1):
                        label = source.get("doc") or f"Source {i}"
                        with st.expander(label):
                            facts = [
                                ("Circular", source.get("circular_no")),
                                ("Date", source.get("date")),
                                ("Section", source.get("section")),
                                ("Page", source.get("page")),
                            ]
                            st.caption(" · ".join(f"{name}: {value}" for name, value in facts if value))
                            st.write(source.get("snippet") or "No excerpt provided.")
                            if source.get("url"):
                                st.markdown(f"[Open source document]({source['url']})")
                elif not result.get("refused"):
                    st.info("No source citations were returned.")

                trace = result.get("trace", {})
                with st.expander("Request details"):
                    st.write(
                        {
                            "refused": result.get("refused", False),
                            "refusal_reason": result.get("refusal_reason"),
                            "cache_hit": trace.get("cache_hit", False),
                            "rewrites_used": trace.get("rewrites_used", 0),
                            "web_fallback_used": trace.get("web_fallback_used", False),
                            "total_latency_ms": round(trace.get("total_latency_ms", 0), 1),
                        }
                    )
            else:
                try:
                    detail = response.json().get("detail", "Request failed.")
                except ValueError:
                    detail = "Request failed."
                st.error(f"API error ({response.status_code}): {detail}")
        except requests.RequestException:
            st.error(f"Could not reach the API at {API_URL}. Start the FastAPI server and try again.")
