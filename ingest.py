import json
import os
import threading
import uuid

from flask import Flask, jsonify, request
from langchain_core.documents import Document
from werkzeug.utils import secure_filename

from rag_pipeline import build_vectorstore_from_documents, chat_answer, rag_answer

ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json"}
DEFAULT_PORT = 8000
MAX_SHORT_TERM_MEMORY_TURNS = 6
DEFAULT_PASTED_TEXT_NAME = "pasted_text.txt"
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Shiva's Rag Application")

app = Flask(__name__)
_session_lock = threading.Lock()
_session_vectorstores = {}
_session_histories = {}


def _build_document(text: str, source: str):
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise ValueError(f"{source} does not contain readable text.")
    return Document(page_content=cleaned_text, metadata={"source": source})


def _decode_document_text(filename: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"{filename} is empty.")

    extension = os.path.splitext(filename)[1].lower()
    decoded = content.decode("utf-8", errors="ignore").strip()
    if not decoded:
        raise ValueError(f"{filename} does not contain readable text.")

    if extension == ".json":
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{filename} is not valid JSON: {exc.msg}.") from exc
        return json.dumps(payload, indent=2, ensure_ascii=False)

    return decoded


def _parse_uploaded_file(uploaded_file):
    filename = secure_filename(uploaded_file.filename or "")
    if not filename:
        raise ValueError("A file without a name was uploaded.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_TEXT_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_TEXT_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type for {filename}. Allowed types: {allowed_list}."
        )

    text = _decode_document_text(filename, uploaded_file.read())
    return _build_document(text, filename)


def _parse_pasted_text(text_content: str, text_name: str):
    source_name = secure_filename((text_name or "").strip()) or DEFAULT_PASTED_TEXT_NAME
    extension = os.path.splitext(source_name)[1].lower()
    if not extension:
        source_name = f"{source_name}.txt"
        extension = ".txt"

    if extension not in ALLOWED_TEXT_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_TEXT_EXTENSIONS))
        raise ValueError(
            f"Unsupported text name {source_name}. Allowed types: {allowed_list}."
        )

    text = (text_content or "").strip()
    if extension == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source_name} is not valid JSON: {exc.msg}.") from exc
        text = json.dumps(payload, indent=2, ensure_ascii=False)

    return _build_document(text, source_name)


def _normalize_intent_text(text: str) -> str:
    normalized = " ".join((text or "").lower().split())
    return normalized.strip("?.! ")


def _memory_response_for_question(question: str, history):
    normalized = _normalize_intent_text(question)
    first_question_markers = (
        "first question",
        "question i asked first",
        "what was my first question",
    )
    previous_question_markers = (
        "previous question",
        "last question",
        "earlier question",
        "what did i ask",
        "question i asked before",
    )
    first_answer_markers = (
        "first answer",
        "what was your first answer",
    )
    previous_answer_markers = (
        "previous answer",
        "last answer",
        "earlier answer",
        "what did you answer",
    )
    identity_markers = (
        "what is your name",
        "what's your name",
        "who are you",
        "tell me your name",
    )
    capability_markers = (
        "what can you do",
        "how can you help",
        "what do you do",
    )

    if any(marker in normalized for marker in identity_markers):
        return (
            f"My name is {ASSISTANT_NAME}. I'm your local assistant for normal chat "
            "and document-grounded question answering."
        )

    if any(marker in normalized for marker in capability_markers):
        return (
            "I can chat in this session, remember recent turns, and answer grounded questions "
            "from uploaded files or pasted text."
        )

    if any(marker in normalized for marker in first_question_markers):
        if not history:
            return "You have not asked an earlier question in this session yet."
        first_question = (history[0].get("question") or "").strip()
        if not first_question:
            return "I couldn't find the first question in this session."
        return f'Your first question in this session was: "{first_question}"'

    if any(marker in normalized for marker in previous_question_markers):
        if not history:
            return "You have not asked a previous question in this session yet."
        previous_question = (history[-1].get("question") or "").strip()
        if not previous_question:
            return "I couldn't find a previous question in this session."
        return f'Your previous question was: "{previous_question}"'

    if any(marker in normalized for marker in first_answer_markers):
        if not history:
            return "I have not given an earlier answer in this session yet."
        first_answer = (history[0].get("answer") or "").strip()
        if not first_answer:
            return "I couldn't find the first answer in this session."
        return f'My first answer in this session was: "{first_answer}"'

    if any(marker in normalized for marker in previous_answer_markers):
        if not history:
            return "I have not given a previous answer in this session yet."
        previous_answer = (history[-1].get("answer") or "").strip()
        if not previous_answer:
            return "I couldn't find a previous answer in this session."
        return f'My previous answer was: "{previous_answer}"'

    return None


def _append_session_history(session_id: str, question: str, answer: str):
    with _session_lock:
        history = _session_histories.setdefault(session_id, [])
        history.append({"question": question, "answer": answer})
        if len(history) > MAX_SHORT_TERM_MEMORY_TURNS:
            del history[:-MAX_SHORT_TERM_MEMORY_TURNS]
        return len(history)


@app.get("/health")
def health():
    with _session_lock:
        count = len(set(_session_vectorstores) | set(_session_histories))
    return jsonify({"status": "ok", "active_sessions": count})


@app.post("/upload")
def upload_documents():
    uploaded_files = request.files.getlist("files")
    text_content = request.form.get("text_content") or ""
    text_name = request.form.get("text_name") or DEFAULT_PASTED_TEXT_NAME
    session_id = (request.form.get("session_id") or "").strip() or str(uuid.uuid4())

    if not uploaded_files and not text_content.strip():
        return (
            jsonify(
                {
                    "error": "Provide files or pasted text before creating an index."
                }
            ),
            400,
        )

    documents = []
    for uploaded_file in uploaded_files:
        try:
            documents.append(_parse_uploaded_file(uploaded_file))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    if text_content.strip():
        try:
            documents.append(_parse_pasted_text(text_content, text_name))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    try:
        vectorstore = build_vectorstore_from_documents(
            documents,
            session_id=session_id,
            reset_collection=True,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Failed to index documents: {exc}"}), 500

    with _session_lock:
        _session_vectorstores[session_id] = vectorstore
        _session_histories.setdefault(session_id, [])

    return jsonify(
        {
            "message": "Documents uploaded and indexed successfully.",
            "session_id": session_id,
            "files_indexed": [doc.metadata.get("source", "") for doc in documents],
            "vector_backend": getattr(vectorstore, "backend_name", "unknown"),
        }
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    session_id = (payload.get("session_id") or "").strip() or str(uuid.uuid4())

    if not question:
        return jsonify({"error": "question is required."}), 400

    with _session_lock:
        history = list(_session_histories.get(session_id, []))

    memory_response = _memory_response_for_question(question, history)
    if memory_response is not None:
        result = {"result": memory_response, "source_documents": []}
    else:
        try:
            result = chat_answer(question, chat_history=history)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": f"Failed to answer question: {exc}"}), 500

    memory_turns = _append_session_history(session_id, question, result.get("result", ""))
    return jsonify(
        {
            "answer": result.get("result", ""),
            "sources": [],
            "source_count": 0,
            "session_id": session_id,
            "memory_turns": memory_turns,
            "vector_backend": "general-chat",
        }
    )


@app.post("/ask")
def ask_question():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    session_id = (payload.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required."}), 400
    if not question:
        return jsonify({"error": "question is required."}), 400

    with _session_lock:
        vectorstore = _session_vectorstores.get(session_id)
        history = list(_session_histories.get(session_id, []))

    if vectorstore is None:
        return (
            jsonify(
                {
                    "error": "Session not found. Index files or pasted text first, or switch to Normal Chat."
                }
            ),
            404,
        )

    memory_response = _memory_response_for_question(question, history)
    if memory_response is not None:
        result = {"result": memory_response, "source_documents": []}
    else:
        try:
            result = rag_answer(question, vectorstore, chat_history=history)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": f"Failed to answer question: {exc}"}), 500

    memory_turns = _append_session_history(session_id, question, result.get("result", ""))
    sources = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in result.get("source_documents", [])
    ]

    return jsonify(
        {
            "answer": result.get("result", ""),
            "sources": sources,
            "source_count": len(sources),
            "session_id": session_id,
            "memory_turns": memory_turns,
            "vector_backend": getattr(vectorstore, "backend_name", "unknown"),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port, debug=False)
