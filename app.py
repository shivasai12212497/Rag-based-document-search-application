import html
import mimetypes
import os
import uuid

import requests
import streamlit as st

st.set_page_config(
    page_title="Codex RAG Desk",
    page_icon=":speech_balloon:",
    layout="wide",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180
SUPPORTED_UPLOAD_TYPES = ["txt", "md", "csv", "log", "json"]
ASSISTANT_MODES = (
    "Normal Chat",
    "Ask From Files",
    "Ask From Pasted Text",
)
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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f2efe8;
        --panel: rgba(255, 255, 255, 0.82);
        --panel-strong: rgba(255, 255, 255, 0.96);
        --border: rgba(25, 27, 31, 0.08);
        --ink: #15171b;
        --muted: #646c77;
        --accent: #0f6d7a;
        --accent-soft: rgba(15, 109, 122, 0.1);
        --user: #1b2733;
        --user-ink: #f6f8fb;
        --shadow: 0 18px 48px rgba(13, 18, 24, 0.08);
    }

    .stApp {
        background:
            radial-gradient(900px 500px at 0% 0%, rgba(15, 109, 122, 0.08), transparent 50%),
            radial-gradient(900px 500px at 100% 0%, rgba(201, 144, 63, 0.10), transparent 45%),
            linear-gradient(180deg, #f6f3ee 0%, var(--bg) 100%);
        color: var(--ink);
        font-family: "Space Grotesk", sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(247, 244, 238, 0.95) 0%, rgba(239, 244, 245, 0.96) 100%);
        border-right: 1px solid var(--border);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.3rem;
        padding-bottom: 6rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .top-shell {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        padding: 20px 22px;
        border-radius: 22px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.86) 0%, rgba(255,255,255,0.72) 100%);
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .top-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }

    .top-subtitle {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.55;
        max-width: 780px;
        margin: 0;
    }

    .mode-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 999px;
        border: 1px solid rgba(15, 109, 122, 0.16);
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        white-space: nowrap;
    }

    .summary-bar {
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.76);
        box-shadow: var(--shadow);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 1rem;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }

    .summary-card {
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--panel-strong);
        padding: 12px 14px;
    }

    .summary-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 6px;
    }

    .summary-value {
        font-size: 14px;
        font-weight: 600;
        line-height: 1.4;
    }

    .message-shell {
        border-radius: 22px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 14px 16px;
        margin-bottom: 10px;
        position: relative;
    }

    .assistant-shell {
        background: var(--panel-strong);
    }

    .user-shell {
        background: var(--user);
        color: var(--user-ink);
        border-color: rgba(255, 255, 255, 0.04);
    }

    .message-role {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .assistant-role {
        color: var(--accent);
        background: var(--accent-soft);
    }

    .user-role {
        color: var(--user-ink);
        background: rgba(255, 255, 255, 0.10);
    }

    .message-text {
        font-size: 15px;
        line-height: 1.7;
        word-wrap: break-word;
    }

    .meta-shell {
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.78);
        padding: 12px 14px;
        box-shadow: var(--shadow);
    }

    .meta-label {
        font-size: 11px;
        color: var(--muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
    }

    .meta-value {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
        line-height: 1.45;
    }

    .welcome-shell {
        border-radius: 24px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(255,255,255,0.74) 100%);
        box-shadow: var(--shadow);
        padding: 26px 28px;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
    }

    .welcome-title {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }

    .welcome-text {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.6;
        margin: 0;
    }

    .hint-card {
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.84);
        padding: 16px 18px;
        min-height: 124px;
    }

    .hint-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent);
        margin-bottom: 8px;
    }

    .hint-body {
        color: var(--ink);
        font-size: 14px;
        line-height: 1.55;
    }

    .stChatInputContainer {
        background: rgba(246, 243, 238, 0.95);
        border-top: 1px solid rgba(21, 23, 27, 0.06);
    }

    .stChatInput textarea {
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.92);
        box-shadow: var(--shadow);
    }

    .stTextInput > div > div > input,
    .stTextArea textarea {
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.92);
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.96);
        color: var(--ink);
        font-weight: 600;
        min-height: 2.7rem;
    }

    .stButton > button:hover {
        border-color: rgba(15, 109, 122, 0.4);
        color: var(--accent);
    }

    @media (max-width: 1100px) {
        .summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .top-shell {
            flex-direction: column;
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
    if remaining > 0:
        return f"{preview}, +{remaining} more"
    return preview


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

    request_kwargs = {
        "data": data,
        "timeout": REQUEST_TIMEOUT_SECONDS,
    }
    if files:
        request_kwargs["files"] = files

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload",
            **request_kwargs,
        )
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

    user_left, user_right = st.columns([1.3, 4], gap="small")
    with user_right:
        st.markdown(
            f"""
            <div class="message-shell user-shell">
                <div class="message-role user-role">You</div>
                <div class="message-text">{question}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    assistant_left, assistant_right = st.columns([4, 1.3], gap="small")
    with assistant_left:
        st.markdown(
            f"""
            <div class="message-shell assistant-shell">
                <div class="message-role assistant-role">Assistant</div>
                <div class="message-text">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if show_sources and sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for index, source in enumerate(sources, start=1):
                    metadata = source.get("metadata") or {}
                    source_name = metadata.get("source", f"Source {index}")
                    st.markdown(f"**{source_name}**")
                    st.write(source.get("content", ""))

    with assistant_right:
        source_count = len(sources)
        st.markdown(
            f"""
            <div class="meta-shell">
                <div class="meta-label">Mode</div>
                <div class="meta-value">{mode}</div>
                <div class="meta-label">Backend</div>
                <div class="meta-value">{backend}</div>
                <div class="meta-label">Sources</div>
                <div class="meta-value">{source_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
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

backend_ok, backend_info = get_backend_health()

with st.sidebar:
    st.title("Codex RAG Desk")
    st.caption("Chat first. Attach knowledge only when you need grounded answers.")

    if backend_ok:
        st.success(
            f"Backend connected | active sessions: {backend_info.get('active_sessions', 0)}"
        )
    else:
        st.error(f"Backend unavailable: {backend_info}")

    st.caption(f"Session: `{st.session_state.session_id[:8]}`")

    st.radio(
        "Conversation mode",
        ASSISTANT_MODES,
        key="assistant_mode",
    )
    st.caption(MODE_DESCRIPTIONS[st.session_state.assistant_mode])

    show_sources = st.checkbox("Show retrieved sources", value=True)

    if st.button("New Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("---")
    st.subheader("Knowledge")

    if st.session_state.assistant_mode == "Ask From Files":
        uploaded_files = st.file_uploader(
            "Upload files",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Supported formats: txt, md, csv, log, json",
        )
        if st.button("Index Files", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one file before indexing.")
            elif not backend_ok:
                st.error("Backend is not reachable.")
            else:
                with st.spinner("Indexing files..."):
                    ok, payload = upload_knowledge(
                        uploaded_files,
                        st.session_state.session_id,
                    )
                if ok:
                    indexed = payload.get("files_indexed", [])
                    st.session_state.session_id = payload.get(
                        "session_id", st.session_state.session_id
                    )
                    st.session_state.knowledge_ready = True
                    st.session_state.knowledge_sources = indexed
                    st.session_state.knowledge_label = _summarize_sources(indexed)
                    st.session_state.knowledge_backend = payload.get("vector_backend")
                    st.success(f"Indexed {len(indexed)} source(s).")
                else:
                    st.error(payload)
    elif st.session_state.assistant_mode == "Ask From Pasted Text":
        text_name = st.text_input(
            "Text name",
            value="notes.txt",
            help="Use a .json name when the pasted content is JSON.",
        )
        pasted_text = st.text_area(
            "Paste source text",
            height=220,
            placeholder="Paste notes, article text, meeting transcript, or JSON here.",
        )
        if st.button("Index Pasted Text", use_container_width=True):
            if not pasted_text.strip():
                st.warning("Paste some text before indexing.")
            elif not backend_ok:
                st.error("Backend is not reachable.")
            else:
                with st.spinner("Indexing pasted text..."):
                    ok, payload = upload_knowledge(
                        [],
                        st.session_state.session_id,
                        text_content=pasted_text,
                        text_name=text_name,
                    )
                if ok:
                    indexed = payload.get("files_indexed", [])
                    st.session_state.session_id = payload.get(
                        "session_id", st.session_state.session_id
                    )
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
        st.success(f"Ready: {st.session_state.knowledge_label}")
        if st.session_state.knowledge_backend:
            st.caption(f"Index backend: {st.session_state.knowledge_backend}")
    else:
        st.caption("No indexed knowledge attached to this session.")

    st.markdown("---")
    st.caption("Supported upload formats: txt, md, csv, log, json")
    st.caption("The first response can still be slower while the model loads.")


current_mode = st.session_state.assistant_mode
knowledge_state = "Ready" if st.session_state.knowledge_ready else "Not indexed"
source_summary = html.escape(st.session_state.knowledge_label)
model_mode_text = html.escape(current_mode)

st.markdown(
    f"""
    <div class="top-shell">
        <div>
            <div class="top-title">Conversation Workspace</div>
            <p class="top-subtitle">
                This layout is chat-first. Keep talking in the main lane, and use the sidebar as your workspace for
                files, pasted text, and session control.
            </p>
        </div>
        <div class="mode-pill">{model_mode_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="summary-bar">
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-label">Mode</div>
                <div class="summary-value">{model_mode_text}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Knowledge</div>
                <div class="summary-value">{knowledge_state}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Indexed Source</div>
                <div class="summary-value">{source_summary}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Backend</div>
                <div class="summary-value">{html.escape(BACKEND_URL)}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.conversation:
    st.markdown(
        """
        <div class="welcome-shell">
            <div class="welcome-title">Start like ChatGPT, switch to RAG when needed.</div>
            <p class="welcome-text">
                Ask normally in Normal Chat, or move to file and pasted-text modes when you want answers grounded in your own material.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hint_one, hint_two, hint_three = st.columns(3)
    with hint_one:
        st.markdown(
            """
            <div class="hint-card">
                <div class="hint-title">Normal Chat</div>
                <div class="hint-body">
                    Use this for general questions, brainstorming, and quick back-and-forth without indexing anything.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hint_two:
        st.markdown(
            """
            <div class="hint-card">
                <div class="hint-title">Files</div>
                <div class="hint-body">
                    Upload txt, md, csv, log, or json files in the sidebar and ask grounded questions from them.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hint_three:
        st.markdown(
            """
            <div class="hint-card">
                <div class="hint-title">Pasted Text</div>
                <div class="hint-body">
                    Paste notes or raw JSON in the sidebar when you want quick grounding without saving a file first.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    for turn in st.session_state.conversation:
        render_turn(turn, show_sources=show_sources)

input_disabled = not backend_ok
if current_mode != "Normal Chat" and not st.session_state.knowledge_ready:
    input_disabled = True

if not backend_ok:
    st.error("Backend is not reachable. Start Flask first.")
elif current_mode != "Normal Chat" and not st.session_state.knowledge_ready:
    st.info("Index files or pasted text in the sidebar before sending a grounded question.")

user_prompt = st.chat_input(
    INPUT_PLACEHOLDERS[current_mode],
    disabled=input_disabled,
)

if user_prompt:
    if current_mode == "Normal Chat":
        request_fn = ask_general_question
        spinner_text = "Generating response..."
    else:
        request_fn = ask_indexed_question
        spinner_text = "Searching indexed knowledge..."

    with st.spinner(spinner_text):
        ok, payload = request_fn(user_prompt.strip(), st.session_state.session_id)

    if ok:
        st.session_state.session_id = payload.get(
            "session_id", st.session_state.session_id
        )
        st.session_state.conversation.append(
            {
                "mode": current_mode,
                "question": user_prompt.strip(),
                "answer": payload.get("answer", ""),
                "sources": payload.get("sources", []),
                "vector_backend": payload.get("vector_backend"),
            }
        )
        st.rerun()
    else:
        st.error(payload)
