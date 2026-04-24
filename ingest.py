import json
import os
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from langchain_core.documents import Document
from werkzeug.utils import secure_filename

from rag_pipeline import build_vectorstore_from_documents, chat_answer, rag_answer

ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json"}
DEFAULT_PORT = 8000
MAX_SHORT_TERM_MEMORY_TURNS = 6
DEFAULT_PASTED_TEXT_NAME = "pasted_text.txt"
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Shiva's Rag Application")

here = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.join(here, "frontend")
static_dir = os.path.join(frontend_dir, "static")

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=static_dir,
)

# Enable CORS for ngrok tunneling
CORS(app)
_session_vectorstores = {}
_session_histories = {}

# 🔥 SIMPLE CACHE (SPEED BOOST)
_query_cache = {}

# ================= HELPERS =================
def _build_document(text: str, source: str):
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise ValueError(f"{source} does not contain readable text.")
    return Document(page_content=cleaned_text, metadata={"source": source})


def _decode_document_text(filename: str, content: bytes) -> str:
    decoded = content.decode("utf-8", errors="ignore").strip()
    if not decoded:
        raise ValueError(f"{filename} does not contain readable text.")

    if filename.endswith(".json"):
        payload = json.loads(decoded)
        return json.dumps(payload, indent=2)

    return decoded


def _parse_uploaded_file(uploaded_file):
    filename = secure_filename(uploaded_file.filename or "")
    if not filename:
        raise ValueError("Invalid file")

    if os.path.splitext(filename)[1].lower() not in ALLOWED_TEXT_EXTENSIONS:
        raise ValueError("Unsupported file type")

    text = _decode_document_text(filename, uploaded_file.read())
    return _build_document(text, filename)


def _parse_pasted_text(text_content: str, text_name: str):
    text = (text_content or "").strip()
    if not text:
        raise ValueError("Empty text")

    return _build_document(text, text_name or DEFAULT_PASTED_TEXT_NAME)


def _append_session_history(session_id: str, question: str, answer: str):
    with _session_lock:
        history = _session_histories.setdefault(session_id, [])
        history.append({"question": question, "answer": answer})
        if len(history) > MAX_SHORT_TERM_MEMORY_TURNS:
            history.pop(0)
        return len(history)

# ================= ROUTES =================
@app.get("/health")
def health():
    return jsonify({"status": "ok", "active_sessions": len(_session_vectorstores)})


@app.post("/upload")
def upload_documents():
    uploaded_files = request.files.getlist("files")
    text_content = request.form.get("text_content") or ""
    text_name = request.form.get("text_name") or DEFAULT_PASTED_TEXT_NAME
    session_id = (request.form.get("session_id") or str(uuid.uuid4())).strip() or str(uuid.uuid4())

    documents = []

    for f in uploaded_files:
        documents.append(_parse_uploaded_file(f))

    if text_content.strip():
        documents.append(_parse_pasted_text(text_content, text_name))

    if not documents:
        return jsonify({"error": "No input provided"}), 400

    print("Documents received:", len(documents))
    vectorstore = build_vectorstore_from_documents(documents)

    with _session_lock:
        _session_vectorstores[session_id] = vectorstore
        _session_histories[session_id] = []

    return jsonify({
        "message": "Indexed successfully",
        "session_id": session_id,
        "files_indexed": [doc.metadata.get("source", "") for doc in documents],
        "files_indexed_count": len(documents),
    })


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    print("Incoming chat query:", question)

    if not question:
        return jsonify({"error": "question required"}), 400

    result = chat_answer(question)

    return jsonify({
        "answer": result["result"],
        "sources": []
    })


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    print("Incoming ask query:", question)
    print("Session id:", session_id)

    if not question or not session_id:
        return jsonify({"error": "missing inputs"}), 400

    # 🔥 CACHE CHECK
    cache_key = (session_id, question)
    if cache_key in _query_cache:
        return jsonify(_query_cache[cache_key])

    with _session_lock:
        vectorstore = _session_vectorstores.get(session_id)
        history = _session_histories.get(session_id, [])

    if not vectorstore:
        return jsonify({"error": "session not found"}), 404

    result = rag_answer(question, vectorstore, chat_history=history)

    _append_session_history(session_id, question, result["result"])

    sources = [
        {
            "content": doc.page_content[:300],  # 🔥 trimmed for UI
            "metadata": doc.metadata
        }
        for doc in result.get("source_documents", [])
    ]

    response = {
        "answer": result["result"],
        "sources": sources,
        "session_id": session_id
    }

    # 🔥 SAVE CACHE
    _query_cache[cache_key] = response

    return jsonify(response)


@app.errorhandler(Exception)
def handle_exception(error):
    import traceback
    traceback.print_exc()
    message = str(error)
    return jsonify({"error": message}), 500


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("frontend/static", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=DEFAULT_PORT, debug=False)