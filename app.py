import os
import uuid

import requests
import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Smart Document Search with RAG",
    page_icon=":books:",
    layout="wide",
)

# ---------------- Theme + Layout ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;500&display=swap');

    :root {
        --bg: #f6f4ef;
        --bg-2: #eef7f7;
        --ink: #1f1f1f;
        --muted: #5b5b5b;
        --card: rgba(255, 255, 255, 0.8);
        --border: rgba(20, 20, 20, 0.08);
        --accent: #1f7a8c;
        --accent-2: #f2a365;
        --shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
    }

    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #fff 0%, var(--bg) 45%, var(--bg-2) 100%);
        color: var(--ink);
        font-family: "Space Grotesk", sans-serif;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: -20% -10% auto auto;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle at 30% 30%, rgba(31, 122, 140, 0.2), rgba(242, 163, 101, 0.12), transparent 60%);
        filter: blur(10px);
        animation: drift 16s ease-in-out infinite;
        z-index: 0;
        pointer-events: none;
    }

    .stApp::after {
        content: "";
        position: fixed;
        inset: auto auto -30% -10%;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle at 60% 40%, rgba(242, 163, 101, 0.2), rgba(31, 122, 140, 0.12), transparent 60%);
        filter: blur(10px);
        animation: drift 18s ease-in-out infinite reverse;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes drift {
        0% { transform: translate(0px, 0px); }
        50% { transform: translate(20px, -16px); }
        100% { transform: translate(0px, 0px); }
    }

    .block-container {
        position: relative;
        z-index: 1;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif;
        letter-spacing: -0.02em;
    }

    .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 26px 28px;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        animation: rise 0.7s ease-out;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(31, 122, 140, 0.18), transparent 60%);
        border-radius: 50%;
    }

    @keyframes rise {
        0% { transform: translateY(10px); opacity: 0.6; }
        100% { transform: translateY(0); opacity: 1; }
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.6fr 1fr;
        gap: 16px;
        align-items: center;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }

    .hero-subtitle {
        font-family: "IBM Plex Serif", serif;
        font-size: 16px;
        color: var(--muted);
        margin: 0;
    }

    .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(31, 122, 140, 0.12);
        color: var(--accent);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .hero-art {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }

    .hero-art svg {
        width: 100%;
        max-width: 240px;
        filter: drop-shadow(0 16px 32px rgba(0, 0, 0, 0.12));
        animation: floaty 6s ease-in-out infinite;
    }

    @keyframes floaty {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        animation: fadein 0.6s ease-out;
    }

    @keyframes fadein {
        0% { transform: translateY(8px); opacity: 0.7; }
        100% { transform: translateY(0); opacity: 1; }
    }

    .callout {
        background: rgba(242, 163, 101, 0.12);
        border: 1px solid rgba(242, 163, 101, 0.35);
        color: #8a4a1d;
        border-radius: 14px;
        padding: 12px 14px;
        font-size: 13px;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid var(--border);
        padding: 12px 14px;
        font-size: 15px;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(31, 122, 140, 0.6);
        box-shadow: 0 0 0 4px rgba(31, 122, 140, 0.12);
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #fff;
        color: var(--ink);
        padding: 8px 12px;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
    }

    .answer {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        font-size: 15px;
        line-height: 1.6;
        animation: glow 1.2s ease-out;
    }

    .answer-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        font-weight: 700;
        font-size: 16px;
    }

    .answer-badge {
        font-size: 12px;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 999px;
        background: rgba(31, 122, 140, 0.15);
        color: var(--accent);
        border: 1px solid rgba(31, 122, 140, 0.25);
    }

    @keyframes glow {
        0% { box-shadow: 0 0 0 rgba(31, 122, 140, 0.0); }
        100% { box-shadow: var(--shadow); }
    }

    .source {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }

    @media (max-width: 900px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-art {
            justify-content: flex-start;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown(
    """
    <div class="hero">
        <div class="pill">RAG SEARCH</div>
        <div class="hero-grid">
            <div>
                <div class="hero-title">Smart Document Search &#x2728;</div>
                <p class="hero-subtitle">Ask questions and get answers grounded in your private documents.</p>
            </div>
            <div class="hero-art" aria-hidden="true">
                <svg viewBox="0 0 220 140" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#1f7a8c" stop-opacity="0.9"/>
                            <stop offset="100%" stop-color="#f2a365" stop-opacity="0.9"/>
                        </linearGradient>
                    </defs>
                    <rect x="10" y="20" rx="16" ry="16" width="200" height="100" fill="url(#g1)"/>
                    <circle cx="55" cy="70" r="20" fill="#ffffff" fill-opacity="0.75"/>
                    <circle cx="110" cy="70" r="26" fill="#ffffff" fill-opacity="0.55"/>
                    <circle cx="165" cy="70" r="18" fill="#ffffff" fill-opacity="0.65"/>
                </svg>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180


def _extract_backend_error(response):
    try:
        payload = response.json()
    except ValueError:
        return f"Backend request failed with status {response.status_code}."
    return payload.get("error") or payload.get("message") or (
        f"Backend request failed with status {response.status_code}."
    )


def get_backend_health():
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=10,
        )
    except requests.RequestException as exc:
        return False, str(exc)

    if not response.ok:
        return False, _extract_backend_error(response)

    return True, response.json()


def upload_documents(uploaded_files, session_id):
    files = [
        (
            "files",
            (uploaded_file.name, uploaded_file.getvalue(), "text/plain"),
        )
        for uploaded_file in uploaded_files
    ]
    data = {"session_id": session_id}

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return False, f"Upload request failed: {exc}"

    if not response.ok:
        return False, _extract_backend_error(response)

    return True, response.json()


def ask_question(question, session_id):
    payload = {"question": question, "session_id": session_id}

    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return False, f"Question request failed: {exc}"

    if not response.ok:
        return False, _extract_backend_error(response)

    return True, response.json()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False
if "last_answer_payload" not in st.session_state:
    st.session_state.last_answer_payload = None

backend_ok, backend_info = get_backend_health()

# ---------------- Sidebar ----------------
st.sidebar.title("About This App")
st.sidebar.markdown("<div class='pill'>DOCS MODE</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    **Smart Document Search** uses:
    - Streamlit Frontend
    - Flask Backend API
    - FAISS Vector Database
    - Retrieval-Augmented Generation
    """
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Backend URL: {BACKEND_URL}")
if backend_ok:
    st.sidebar.success(
        f"Backend connected (active sessions: {backend_info.get('active_sessions', 0)})"
    )
else:
    st.sidebar.error(f"Backend unavailable: {backend_info}")

# ---------------- Main Content ----------------
left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("1) Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=["txt", "md", "csv", "log"],
        accept_multiple_files=True,
        help="Upload one or more text files to build your searchable knowledge base.",
    )

    upload_clicked = st.button("Index Uploaded Files")
    if upload_clicked:
        if not uploaded_files:
            st.warning("Please select at least one file.")
        elif not backend_ok:
            st.error("Backend is not reachable. Start Flask backend first.")
        else:
            with st.spinner("Uploading and indexing documents..."):
                ok, payload = upload_documents(
                    uploaded_files,
                    st.session_state.session_id,
                )
            if ok:
                st.session_state.session_id = payload.get(
                    "session_id", st.session_state.session_id
                )
                st.session_state.documents_ready = True
                st.session_state.last_answer_payload = None
                indexed = payload.get("files_indexed", [])
                st.success(
                    f"Indexed {len(indexed)} file(s): {', '.join(indexed) if indexed else 'N/A'}"
                )
            else:
                st.error(payload)

    st.caption(f"Session ID: `{st.session_state.session_id}`")

    st.subheader("2) Ask a Question :speech_balloon:")
    with st.form("question_form", clear_on_submit=False):
        query = st.text_input(
            "Type your question",
            placeholder="Example: What is the data science lifecycle?",
        )
        submit_question = st.form_submit_button("Generate Answer")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Quality Guardrails :white_check_mark:")
    st.markdown(
        """
        Upload files first, then ask document-grounded questions.
        If the answer is not in your uploaded context, the assistant will say so.
        """
    )
    st.markdown(
        "<div class='callout'>Tip: Upload clean, specific content and ask focused questions for better accuracy.</div>",
        unsafe_allow_html=True,
    )
    show_sources = st.checkbox("Show sources by default", value=True)
    if st.session_state.documents_ready:
        st.success("Documents are indexed and ready.")
    else:
        st.warning("No indexed documents yet.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ---------------- Query Execution ----------------
if submit_question:
    if not query.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.documents_ready:
        st.warning("Upload and index documents before asking questions.")
    elif not backend_ok:
        st.error("Backend is not reachable. Start Flask backend first.")
    else:
        with st.spinner("Searching uploaded documents and generating answer..."):
            ok, payload = ask_question(query.strip(), st.session_state.session_id)
        if ok:
            st.session_state.last_answer_payload = payload
        else:
            st.session_state.last_answer_payload = None
            st.error(payload)

result_payload = st.session_state.last_answer_payload
if result_payload:
    sources = result_payload.get("sources", [])
    has_sources = bool(sources)

    if show_sources:
        tabs = st.tabs(["Answer", "Sources"])
    else:
        tabs = [st.container()]

    with tabs[0]:
        st.markdown("<div class='answer'>", unsafe_allow_html=True)
        if has_sources:
            st.markdown(
                "<div class='answer-title'>Answer found <span class='answer-badge'>RAG VERIFIED</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Matched {len(sources)} supporting chunks from your uploads.")
        else:
            st.markdown(
                "<div class='answer-title'>No exact answer <span class='answer-badge'>LIMITED</span></div>",
                unsafe_allow_html=True,
            )
        st.write(result_payload.get("answer", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    if show_sources:
        with tabs[1]:
            if sources:
                for i, source in enumerate(sources, start=1):
                    content = source.get("content", "")
                    st.markdown(
                        f"<div class='source'><strong>Source {i}</strong><br>{content}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No sources to show for this answer.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Powered by Flask + Streamlit + RAG.")
