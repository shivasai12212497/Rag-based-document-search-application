import os
import subprocess
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# CONFIG 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 120          
CHUNK_OVERLAP = 20        
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_FALLBACKS = (
    "llama3",
    "llama3.2:3b",
    "gemma3:1b",
)

# ================= EMBEDDINGS =================
@lru_cache(maxsize=1)
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model: {exc}") from exc


# ================= LLM =================
@lru_cache(maxsize=1)
def _list_installed_ollama_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    installed = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            installed.append(parts[0])
    return installed


def _resolve_ollama_model():
    installed = _list_installed_ollama_models()

    for model_name in (OLLAMA_MODEL, *OLLAMA_MODEL_FALLBACKS):
        if model_name in installed:
            return model_name

    raise RuntimeError("No suitable Ollama model found")


@lru_cache(maxsize=1)
def get_llm():
    return Ollama(model=_resolve_ollama_model(), base_url=OLLAMA_BASE_URL)


def _invoke_llm(prompt: str) -> str:
    llm = get_llm()
    return llm.invoke(prompt[:3000])   # 🔥 prevent overload


# ================= SPLITTING =================
def split_documents(documents):
    documents = [doc for doc in documents if doc.page_content.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


# ================= VECTORSTORE =================
def build_vectorstore_from_documents(documents, session_id=None, reset_collection=False):
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No valid content found")

    db = FAISS.from_documents(chunks, get_embeddings())
    db.backend_name = "FAISS"
    return db


# ================= RAG =================
def rag_answer(question, vectorstore, chat_history=None, top_k=3):

    q = question.lower()

    # 🔥 dynamic retrieval (VERY IMPORTANT)
    if "highest" in q or "maximum" in q or "lowest" in q:
        docs = vectorstore.similarity_search(question, k=15)
    else:
        docs = vectorstore.similarity_search(question, k=5)

    if not docs:
        return {
            "result": "I couldn't find relevant information in the documents.",
            "source_documents": []
        }

    context = "\n\n".join([d.page_content for d in docs])

    history_text = ""
    if chat_history:
        for turn in chat_history[-3:]:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    # 🔥 IMPROVED PROMPT
    prompt = f"""
You are a data analyst.

Carefully read ALL context and compare values before answering.

If question asks for highest/lowest:
→ Compare all entries before answering.

If answer not found:
→ Say "I don't have enough information in the documents."

Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    answer = _invoke_llm(prompt)

    return {
        "result": answer.strip(),
        "source_documents": docs
    }


# ================= CHAT =================
def chat_answer(question, chat_history=None):
    history_text = ""

    if chat_history:
        for turn in chat_history[-5:]:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""
You are a helpful assistant.

Conversation:
{history_text}

User: {question}
Assistant:
"""

    answer = _invoke_llm(prompt)

    return {
        "result": answer.strip(),
        "source_documents": []
    }