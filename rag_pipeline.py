import os
import subprocess
import random
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
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


# ================= LLM =================
@lru_cache(maxsize=1)
def _list_installed_ollama_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return [line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()]


def _resolve_ollama_model():
    installed = _list_installed_ollama_models()
    for model_name in (OLLAMA_MODEL, *OLLAMA_MODEL_FALLBACKS):
        if model_name in installed:
            return model_name
    raise RuntimeError("No suitable Ollama model found")


@lru_cache(maxsize=1)
def get_llm():
    return Ollama(
        model=_resolve_ollama_model(),
        base_url=OLLAMA_BASE_URL,
        temperature=0.7   
    )


def _invoke_llm(prompt: str) -> str:
    return get_llm().invoke(prompt[:3000])


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

    # 🔥 better retrieval
    if any(x in q for x in ["highest", "maximum", "lowest", "compare"]):
        docs = vectorstore.similarity_search(question, k=12)
    else:
        docs = vectorstore.similarity_search(question, k=6)

    if not docs:
        return {
            "result": "I couldn't find relevant information in the documents.",
            "source_documents": []
        }

    # 🔥 shuffle → avoid repetition
    random.shuffle(docs)

    context = "\n\n".join([d.page_content for d in docs])

    # 🔥 reduce memory (avoid loops)
    history_text = ""
    if chat_history:
        for turn in chat_history[-2:]:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""
You are a smart data assistant.

IMPORTANT:
- Respond ONLY in the same language as the user's question.
- Do NOT switch language.
- Do NOT mix languages.

- Read ALL context carefully
- Do NOT repeat previous answers
- Answer clearly and naturally
- Compare values if needed

If answer is not in context:
Say: "I don't have enough information in the documents."

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
        for turn in chat_history[-2:]:   # 🔥 reduce memory loop
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""
You are a helpful assistant.

- Give natural answers
- Avoid repeating same sentences
- Be concise and clear

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