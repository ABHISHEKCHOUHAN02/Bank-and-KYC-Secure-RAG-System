"""Streamlit client for the BankGuard RAG FastAPI service."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("BANKGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="BankGuard RAG", page_icon="🏦", layout="centered")
st.title("🏦 BankGuard RAG")
view = st.sidebar.radio("Workspace", ["Ask a question", "Manage documents"])

if view == "Ask a question":
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
else:
    st.caption("Upload PDF, TXT, or Markdown sources and remove documents you uploaded.")
    st.warning("Each document change rebuilds the full search index. The original bundled documents cannot be deleted here.")
    admin_token = st.text_input("Document admin token", type="password", key="doc_admin_token")
    headers = {"X-Admin-Token": admin_token} if admin_token else {}

    with st.form("upload_document_form", clear_on_submit=True):
        source_type = st.selectbox("Document collection", ["rbi", "fictional_bank"])
        uploaded = st.file_uploader("Select a PDF, TXT, or Markdown file", type=["pdf", "txt", "md"])
        upload_submitted = st.form_submit_button("Upload and reindex", type="primary")
    if upload_submitted:
        if not admin_token:
            st.error("Enter the document admin token first.")
        elif uploaded is None:
            st.warning("Choose a file to upload.")
        else:
            try:
                with st.spinner("Saving the document and rebuilding all indexes…"):
                    response = requests.post(
                        f"{API_URL}/documents",
                        headers=headers,
                        data={"source_type": source_type},
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=(5, 900),
                    )
                if response.ok:
                    st.success(response.json().get("message", "Document indexed."))
                    st.rerun()
                else:
                    st.error(f"Upload failed ({response.status_code}): {response.json().get('detail', 'Request failed.')}")
            except requests.RequestException:
                st.error(f"Could not reach the API at {API_URL}.")

    st.subheader("Indexed documents")
    if not admin_token:
        st.info("Set DOCS_ADMIN_TOKEN in the API environment, then enter the same token here to manage documents.")
    else:
        try:
            response = requests.get(f"{API_URL}/documents", headers=headers, timeout=(5, 30))
            if response.ok:
                for doc in response.json():
                    title = f"{doc['filename']} · {doc['source_type']} · {doc['size_bytes']:,} bytes"
                    with st.expander(title):
                        if doc["managed_upload"]:
                            confirm = st.checkbox("Confirm removal", key=f"confirm_{doc['source_type']}_{doc['filename']}")
                            if st.button("Remove and reindex", key=f"delete_{doc['source_type']}_{doc['filename']}",
                                         disabled=not confirm):
                                try:
                                    with st.spinner("Removing the document and rebuilding all indexes…"):
                                        deleted = requests.delete(
                                            f"{API_URL}/documents/{doc['source_type']}/{requests.utils.quote(doc['filename'], safe='')}",
                                            headers=headers,
                                            timeout=(5, 900),
                                        )
                                    if deleted.ok:
                                        st.success(deleted.json().get("message", "Document removed."))
                                        st.rerun()
                                    else:
                                        st.error(f"Removal failed ({deleted.status_code}): {deleted.json().get('detail', 'Request failed.')}")
                                except requests.RequestException:
                                    st.error(f"Could not reach the API at {API_URL}.")
                        else:
                            st.caption("Bundled source document — read-only in this interface.")
            else:
                try:
                    detail = response.json().get("detail", "Request failed.")
                except ValueError:
                    detail = "Request failed."
                st.error(f"Could not list documents ({response.status_code}): {detail}")
        except requests.RequestException:
            st.error(f"Could not reach the API at {API_URL}. Start the FastAPI server and try again.")
