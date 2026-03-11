import os
import threading
import uuid

from flask import Flask, jsonify, request
from langchain_core.documents import Document
from werkzeug.utils import secure_filename

from rag_pipeline import build_vectorstore_from_documents, rag_answer

ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log"}
DEFAULT_PORT = 8000

app = Flask(__name__)
_session_lock = threading.Lock()
_session_vectorstores = {}


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

    content = uploaded_file.read()
    if not content:
        raise ValueError(f"{filename} is empty.")

    text = content.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"{filename} does not contain readable text.")

    return Document(page_content=text, metadata={"source": filename})


@app.get("/health")
def health():
    with _session_lock:
        count = len(_session_vectorstores)
    return jsonify({"status": "ok", "active_sessions": count})


@app.post("/upload")
def upload_documents():
    uploaded_files = request.files.getlist("files")
    session_id = (request.form.get("session_id") or "").strip() or str(uuid.uuid4())

    if not uploaded_files:
        return jsonify({"error": "No files provided. Upload at least one file."}), 400

    documents = []
    for uploaded_file in uploaded_files:
        try:
            documents.append(_parse_uploaded_file(uploaded_file))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    try:
        vectorstore = build_vectorstore_from_documents(documents)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _session_lock:
        _session_vectorstores[session_id] = vectorstore

    return jsonify(
        {
            "message": "Documents uploaded and indexed successfully.",
            "session_id": session_id,
            "files_indexed": [doc.metadata.get("source", "") for doc in documents],
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

    if vectorstore is None:
        return (
            jsonify(
                {
                    "error": "Session not found. Upload files first to create an index."
                }
            ),
            404,
        )

    result = rag_answer(question, vectorstore)
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
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port, debug=False)
