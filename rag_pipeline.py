import os
import subprocess
import random
from functools import lru_cache
import statistics

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# ================= CONFIG =================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ================= EMBEDDINGS =================
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

# ================= LLM =================
@lru_cache(maxsize=1)
def _list_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return [line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()]

def _resolve_model():
    models = _list_models()
    for m in ["llama3", "llama3.2:3b", "gemma3:1b"]:
        if m in models:
            return m
    raise RuntimeError("No model found")

@lru_cache(maxsize=1)
def get_llm():
    return Ollama(
        model=_resolve_model(),
        base_url=OLLAMA_BASE_URL,
        temperature=0.6
    )

def _invoke(prompt):
    return get_llm().invoke(prompt[:3500])

# ================= SPLIT =================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# ================= VECTORSTORE =================
def build_vectorstore_from_documents(documents, session_id=None, reset_collection=False):
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No valid content")
    return FAISS.from_documents(chunks, get_embeddings())

# ================= 🔥 DATA EXTRACTION =================
def extract_data(vectorstore):
    docs = vectorstore.similarity_search("", k=100)

    records = []
    seen = set()

    for doc in docs:
        record = {}
        for line in doc.page_content.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                record[k.strip()] = v.strip()

        if record:
            key = str(record)
            if key not in seen:
                seen.add(key)
                records.append(record)

    return records

# ================= 🔥 AUTO ANALYTICS =================
def compute_analytics(question, vectorstore):
    data = extract_data(vectorstore)
    q = question.lower()

    if not data:
        return None

    # detect numeric columns automatically
    numeric_keys = []
    for key in data[0]:
        try:
            int(data[0][key])
            numeric_keys.append(key)
        except:
            pass

    if not numeric_keys:
        return None

    key = numeric_keys[0]  # pick first numeric column
    values = [int(d[key]) for d in data if key in d]

    if not values:
        return None

    # 🔥 operations
    if "mean" in q or "average" in q:
        return f"Average {key} is {sum(values)/len(values):.2f}"

    if "median" in q:
        return f"Median {key} is {statistics.median(values)}"

    if "mode" in q:
        try:
            return f"Mode {key} is {statistics.mode(values)}"
        except:
            return "No unique mode found"

    if "highest" in q or "maximum" in q:
        best = max(data, key=lambda x: int(x.get(key, 0)))
        return f"{best.get('Name','Record')} has highest {key}: {best.get(key)}"

    if "lowest" in q or "minimum" in q:
        low = min(data, key=lambda x: int(x.get(key, 999999)))
        return f"{low.get('Name','Record')} has lowest {key}: {low.get(key)}"

    if "difference" in q or "diff" in q:
        return f"Difference between highest and lowest {key} is {max(values) - min(values)}"

    if "count" in q or "total" in q:
        return f"Total records: {len(data)}"

    return None

# ================= RAG =================
def rag_answer(question, vectorstore, chat_history=None):

    # 🔥 AUTO ANALYTICS FIRST
    computed = compute_analytics(question, vectorstore)
    if computed:
        return {
            "result": computed,
            "source_documents": []
        }

    # 🔥 RAG fallback
    docs = vectorstore.similarity_search(question, k=10)

    if not docs:
        return {
            "result": "I couldn't find relevant information.",
            "source_documents": []
        }

    context = "\n\n".join([d.page_content for d in docs])

    history_text = ""
    if chat_history:
        for turn in chat_history[-3:]:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""
You are an intelligent assistant working on user-provided data.

STRICT RULES:
- Answer ONLY using the given context
- Do NOT hallucinate

SPECIAL:
- If user asks for details → give full structured answer

LANGUAGE:
- Respond in same language

Conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    answer = _invoke(prompt)

    return {
        "result": answer.strip(),
        "source_documents": docs
    }

# ================= CHAT =================
def chat_answer(question, chat_history=None):

    history_text = ""
    if chat_history:
        for turn in chat_history[-3:]:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""
You are a helpful assistant.

- Respond naturally
- Match user language

Conversation:
{history_text}

User: {question}
Assistant:
"""

    answer = _invoke(prompt)

    return {
        "result": answer.strip(),
        "source_documents": []
    }