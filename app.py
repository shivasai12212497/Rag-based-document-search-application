import html
import mimetypes
import os
import uuid

import requests
import streamlit as st

st.set_page_config(
    page_title="Shiva's RAG Studio",
    page_icon="🎯",
    layout="wide",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180
SUPPORTED_UPLOAD_TYPES = ["txt", "md", "csv", "log", "json"]
ASSISTANT_MODES = ["Normal Chat", "Ask From Files", "Ask From Pasted Text"]
MODE_DESCRIPTIONS = {
    "Normal Chat": "Direct assistant replies without retrieval.",
    "Ask From Files": "Ground answers in uploaded files for this session.",
    "Ask From Pasted Text": "Ground answers in pasted notes or JSON for this session.",
}
INPUT_PLACEHOLDERS = {
    "Normal Chat": "Message the assistant...",
    "Ask From Files": "Ask about the indexed files...",
    "Ask From Pasted Text": "Ask about the indexed pasted text...",
}


def _inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            color-scheme: light;
            font-family: 'Inter', sans-serif;
            background: #eef2ff;
            color: #111827;
        }

        .stApp {
            background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 60%, #ffffff 100%);
        }

        .block-container {
            padding: 1.5rem 2rem 2rem 2rem;
            max-width: 1440px;
        }

        .page-header,
        .section-card,
        .hint-card,
        .sidebar-card {
            border-radius: 24px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.06);
        }

        .page-header {
            padding: 28px;
            margin-bottom: 22px;
        }

        .page-title {
            margin: 0;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #0f172a;
        }

        .page-subtitle {
            margin: 0.85rem 0 0;
            color: #475569;
            line-height: 1.8;
            max-width: 780px;
            font-size: 1rem;
        }

        .tag-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.8rem 1rem;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.14);
            color: #4338ca;
            font-size: 0.92rem;
            font-weight: 700;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.25rem;
        }

        .metric-card {
            padding: 1rem 1rem;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.06);
            min-height: 100px;
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #64748b;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0f172a;
        }

        .chat-message {
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
            background: #ffffff;
        }

        .chat-message.user {
            border-left: 4px solid #4338ca;
            background: rgba(67, 56, 202, 0.06);
        }

        .chat-message.assistant {
            border-left: 4px solid #0f766e;
            background: rgba(15, 118, 110, 0.06);
        }

        .chat-role {
            font-size: 0.83rem;
            font-weight: 700;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.75rem;
        }

        .chat-text {
            margin: 0;
            font-size: 1rem;
            line-height: 1.75;
            color: #1f2937;
        }

        .chat-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1rem;
        }

        .chat-badge {
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.05);
            color: #475569;
            font-size: 0.82rem;
            padding: 0.68rem 0.9rem;
        }

        .hint-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .hint-card {
            padding: 1.2rem;
            border-radius: 20px;
            background: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .hint-card h3 {
            margin: 0 0 0.65rem;
            font-size: 1rem;
            color: #0f172a;
        }

        .hint-card p {
            margin: 0;
            color: #475569;
            line-height: 1.8;
            font-size: 0.95rem;
        }

        .sidebar-card {
            padding: 1.1rem 1.15rem;
            margin-bottom: 1rem;
        }

        .sidebar-card h4 {
            margin: 0 0 0.75rem;
            font-size: 1rem;
            color: #0f172a;
        }

        .sidebar-note {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.7;
        }

        .stButton > button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #4f46e5, #0f766e);
            color: white;
            font-weight: 700;
            min-height: 3rem;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stTextArea textarea,
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stFileUploader {
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.12);
            background: #ffffff;
        }

        @media (max-width: 1040px) {
            .metric-row,
            .hint-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 720px) {
            .page-header,
            .metric-row,
            .hint-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _extract_backend_error(response):
    try:
        payload = response.json()
    except ValueError:
        return f"Backend request failed with status {response.status_code}."
    return payload.get("error") or payload.get("message") or (
        f"Backend request failed with status {response.status_code}."
    )


def _guess_content_type(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "text/plain"


def _safe_html(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def _summarize_sources(sources):
    if not sources:
        return "No indexed source"
    if len(sources) == 1:
        return sources[0]
    preview = ", ".join(sources[:2])
    remaining = len(sources) - 2
    return f"{preview}, +{remaining} more" if remaining else preview


def get_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
    except requests.RequestException as exc:
        return False, str(exc)
    if not response.ok:
        return False, _extract_backend_error(response)
    return True, response.json()


def upload_knowledge(uploaded_files, session_id, text_content="", text_name="notes.txt"):
    files = []
    for uploaded_file in uploaded_files or []:
        files.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    _guess_content_type(uploaded_file.name),
                ),
            )
        )
    data = {"session_id": session_id}
    if text_content.strip():
        data["text_content"] = text_content
        data["text_name"] = text_name
    request_kwargs = {"data": data, "timeout": REQUEST_TIMEOUT_SECONDS}
    if files:
        request_kwargs["files"] = files
    try:
        response = requests.post(f"{BACKEND_URL}/upload", **request_kwargs)
    except requests.RequestException as exc:
        return False, f"Upload request failed: {exc}"
    if not response.ok:
        return False, _extract_backend_error(response)
    return True, response.json()


def ask_general_question(question, session_id):
    payload = {"question": question, "session_id": session_id}
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return False, f"Chat request failed: {exc}"
    if not response.ok:
        return False, _extract_backend_error(response)
    return True, response.json()


def ask_indexed_question(question, session_id):
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


def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.knowledge_ready = False
    st.session_state.knowledge_label = "No indexed source"
    st.session_state.knowledge_backend = None
    st.session_state.knowledge_sources = []
    st.session_state.conversation = []


def render_turn(turn, show_sources):
    question = _safe_html(turn.get("question", ""))
    answer = _safe_html(turn.get("answer", ""))
    mode = html.escape(turn.get("mode", ""))
    backend = html.escape(turn.get("vector_backend") or "local")
    sources = turn.get("sources", [])
    source_count = len(sources)
    st.markdown(
        f"""
        <div class="chat-message user">
            <div class="chat-role">You</div>
            <div class="chat-text">{question}</div>
            <div class="chat-badge-row">
                <div class="chat-badge">Mode: {mode}</div>
                <div class="chat-badge">Sources: {source_count}</div>
            </div>
        </div>
        <div class="chat-message assistant">
            <div class="chat-role">Assistant</div>
            <div class="chat-text">{answer}</div>
            <div class="chat-badge-row">
                <div class="chat-badge">Backend: {backend}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_sources and sources:
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            source_name = metadata.get("source", f"Source {index}")
            st.markdown(
                f"""
                <div class="section-card" style="margin-bottom: 1rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.9rem;">
                        <div style="font-size:1rem;font-weight:700;color:#0f172a;">Source {index}</div>
                        <div class="chat-badge">{html.escape(source_name)}</div>
                    </div>
                    <div style="color:#475569;line-height:1.7;font-size:0.95rem;">{html.escape(source.get('content', ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if "session_id" not in st.session_state:
    reset_session()

if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = ASSISTANT_MODES[0]
if "knowledge_ready" not in st.session_state:
    st.session_state.knowledge_ready = False
if "knowledge_label" not in st.session_state:
    st.session_state.knowledge_label = "No indexed source"
if "knowledge_backend" not in st.session_state:
    st.session_state.knowledge_backend = None
if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = []
if "conversation" not in st.session_state:
    st.session_state.conversation = []

_inject_style()
backend_ok, backend_info = get_backend_health()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Session controls</h4>
            <p class="sidebar-note">Choose your mode, attach knowledge if needed, and start the conversation below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if backend_ok:
        st.success(f"Backend online • active sessions: {backend_info.get('active_sessions', 0)}")
    else:
        st.error(f"Backend unavailable: {backend_info}")

    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    st.radio("Conversation mode", ASSISTANT_MODES, key="assistant_mode")
    st.caption(MODE_DESCRIPTIONS[st.session_state.assistant_mode])
    show_sources = st.checkbox("Show retrieved sources", value=True)

    if st.button("New session", use_container_width=True):
        reset_session()

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Knowledge builder</h4>
            <p class="sidebar-note">Upload files or paste text here for grounded answers when using the selected mode.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.assistant_mode == "Ask From Files":
        uploaded_files = st.file_uploader(
            "Upload files",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Supported formats: txt, md, csv, log, json",
        )
        if st.button("Index files", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one file before indexing.")
            elif not backend_ok:
                st.error("Backend is not reachable.")
            else:
                with st.spinner("Indexing files..."):
                    ok, payload = upload_knowledge(uploaded_files, st.session_state.session_id)
                if ok:
                    indexed = payload.get("files_indexed", [])
                    st.session_state.session_id = payload.get("session_id", st.session_state.session_id)
                    st.session_state.knowledge_ready = True
                    st.session_state.knowledge_sources = indexed
                    st.session_state.knowledge_label = _summarize_sources(indexed)
                    st.session_state.knowledge_backend = payload.get("vector_backend")
                    st.success(f"Indexed {len(indexed)} source(s).")
                else:
                    st.error(payload)
    elif st.session_state.assistant_mode == "Ask From Pasted Text":
        text_name = st.text_input("Text name", value="notes.txt", help="Use a .json name when pasting JSON.")
        pasted_text = st.text_area("Paste source text", height=220, placeholder="Paste notes or JSON here.")
        if st.button("Index pasted text", use_container_width=True):
            if not pasted_text.strip():
                st.warning("Paste some text before indexing.")
            elif not backend_ok:
                st.error("Backend is not reachable.")
            else:
                with st.spinner("Indexing pasted text..."):
                    ok, payload = upload_knowledge([], st.session_state.session_id, text_content=pasted_text, text_name=text_name)
                if ok:
                    indexed = payload.get("files_indexed", [])
                    st.session_state.session_id = payload.get("session_id", st.session_state.session_id)
                    st.session_state.knowledge_ready = True
                    st.session_state.knowledge_sources = indexed
                    st.session_state.knowledge_label = _summarize_sources(indexed)
                    st.session_state.knowledge_backend = payload.get("vector_backend")
                    st.success(f"Indexed {len(indexed)} source(s).")
                else:
                    st.error(payload)
    else:
        st.info("Normal Chat does not require indexing.")

    if st.session_state.knowledge_ready:
        st.success(f"Knowledge attached: {st.session_state.knowledge_label}")
        if st.session_state.knowledge_backend:
            st.caption(f"Index backend: {st.session_state.knowledge_backend}")
    else:
        st.caption("No indexed knowledge attached to this session.")

    st.markdown("---")
    st.caption("Supported formats: txt, md, csv, log, json")
    st.caption("First response may take longer while the backend warms up.")

current_mode = st.session_state.assistant_mode
knowledge_state = "Ready" if st.session_state.knowledge_ready else "Not indexed"
source_summary = html.escape(st.session_state.knowledge_label)
model_mode_text = html.escape(current_mode)

st.markdown(
    f"""
    <div class="page-header">
        <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:start;">
            <div>
                <h1 class="page-title">Shiva's RAG Studio</h1>
                <p class="page-subtitle">A cleaner knowledge chat interface built for fast, grounded responses. Attach files or paste text only when you need stronger quotes and references.</p>
            </div>
            <div class="tag-pill">Smart chat with knowledge support</div>
        </div>
        <div class="metric-row">
            <div class="metric-card"><div class="metric-label">Mode</div><div class="metric-value">{model_mode_text}</div></div>
            <div class="metric-card"><div class="metric-label">Knowledge</div><div class="metric-value">{knowledge_state}</div></div>
            <div class="metric-card"><div class="metric-label">Indexed source</div><div class="metric-value">{source_summary}</div></div>
            <div class="metric-card"><div class="metric-label">Backend</div><div class="metric-value">{html.escape(BACKEND_URL)}</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([3, 1], gap="large")

with main_col:
    conversation_placeholder = st.container()
    with conversation_placeholder:
        if not st.session_state.conversation:
            st.markdown(
                """
                <div class="section-card">
                    <h3 style="margin:0 0 0.75rem; color:#0f172a;">Ready to ask your first question</h3>
                    <p style="margin:0; color:#475569; line-height:1.8;">Choose a mode, attach knowledge if needed, then ask your question using the chat box below. The assistant will keep the conversation flowing in the main pane.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="hint-grid">
                    <div class="hint-card"><h3>Normal Chat</h3><p>Use this for general questions, brainstorming, and everyday prompts without indexing anything.</p></div>
                    <div class="hint-card"><h3>Ask From Files</h3><p>Upload your documents to ground answers in your own reports, notes, and CSV content.</p></div>
                    <div class="hint-card"><h3>Ask From Pasted Text</h3><p>Paste raw notes, meeting text, or JSON for a quick knowledge grounding workflow.</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for turn in st.session_state.conversation:
                render_turn(turn, show_sources=show_sources)

    st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)

    user_prompt = st.chat_input(
        INPUT_PLACEHOLDERS[current_mode],
        disabled=not backend_ok or (current_mode != "Normal Chat" and not st.session_state.knowledge_ready),
    )

    if user_prompt:
        request_fn = ask_general_question if current_mode == "Normal Chat" else ask_indexed_question
        spinner_text = "Generating response..." if current_mode == "Normal Chat" else "Searching indexed knowledge..."
        with st.spinner(spinner_text):
            ok, payload = request_fn(user_prompt.strip(), st.session_state.session_id)
        if ok:
            st.session_state.session_id = payload.get("session_id", st.session_state.session_id)
            st.session_state.conversation.append(
                {
                    "mode": current_mode,
                    "question": user_prompt.strip(),
                    "answer": payload.get("answer", ""),
                    "sources": payload.get("sources", []),
                    "vector_backend": payload.get("vector_backend"),
                }
            )
            conversation_placeholder.empty()
            with conversation_placeholder:
                for turn in st.session_state.conversation:
                    render_turn(turn, show_sources=show_sources)
        else:
            st.error(payload)

with side_col:
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Quick actions</h4>
            <p class="sidebar-note">Switch modes and index knowledge here. Keep the conversation focused in the main pane.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>Tips</h4>
            <ul style="margin:0; padding-left:1.2rem; color:#475569;">
                <li>Keep questions clear and concise.</li>
                <li>Use files for documents that should be referenced directly.</li>
                <li>Paste text for quick one-off knowledge checks.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not backend_ok:
    st.error("Backend is not reachable. Start the backend service before using the app.")
elif current_mode != "Normal Chat" and not st.session_state.knowledge_ready:
    st.info("Index files or pasted text in the sidebar before asking a grounded question.")
