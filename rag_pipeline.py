import csv
import io
import json
import os
import re
import statistics
import subprocess
from functools import lru_cache

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
def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (key or "").lower()).strip()


def _parse_json_records(text: str):
    try:
        payload = json.loads(text)
    except Exception:
        return []

    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return [dict(item) for item in payload]

    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                return [dict(item) for item in value]
    return []


def _parse_csv_records(text: str):
    try:
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    except Exception:
        return []

    if len(rows) < 2:
        return []

    header = [cell.strip() for cell in rows[0]]
    records = []
    for row in rows[1:]:
        if len(row) < len(header):
            continue
        record = {header[i]: row[i].strip() for i in range(len(header))}
        if record:
            records.append(record)
    return records


def _parse_key_value_blocks(text: str):
    records = []
    for block in re.split(r"\n\s*\n", text.strip()):
        record = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()
        if record:
            records.append(record)
    return records


def extract_data(vectorstore):
    docs = vectorstore.similarity_search("", k=100)
    records = []
    seen = set()

    for doc in docs:
        text = doc.page_content.strip()
        doc_records = _parse_json_records(text)
        if not doc_records:
            doc_records = _parse_key_value_blocks(text)
        if not doc_records:
            doc_records = _parse_csv_records(text)

        for record in doc_records:
            if not record:
                continue
            normalized = tuple(sorted((k, v) for k, v in record.items()))
            if normalized in seen:
                continue
            seen.add(normalized)
            records.append(record)

    return records


def _numeric_value(value):
    if value is None:
        return None
    try:
        cleaned = str(value).replace(",", "").strip()
        return float(cleaned) if "." in cleaned else int(cleaned)
    except Exception:
        return None


def _score_numeric_field(field_name: str, question: str):
    field = field_name.lower()
    score = 0
    if any(term in field for term in ["salary", "wage", "pay", "compensation", "amount", "price", "cost"]):
        score += 20
    if any(term in field for term in ["age", "years", "ageyears"]):
        score += 18
    if any(term in field for term in ["score", "rating", "rank", "points"]):
        score += 16
    if any(term in field for term in ["id", "code", "number"]):
        score -= 5
    if any(term in question for term in _normalize_key(field).split()):
        score += 12
    if "salary" in question and "salary" in field:
        score += 25
    if "age" in question and "age" in field:
        score += 25
    return score


def _pick_numeric_field(question: str, data):
    numeric_fields = []
    for key in data[0].keys():
        values = [_numeric_value(item.get(key)) for item in data]
        if any(v is not None for v in values):
            numeric_fields.append((key, [v for v in values if v is not None]))

    if not numeric_fields:
        return None, None

    question_lower = question.lower()
    if any(term in question_lower for term in ["count", "total", "records"]):
        return None, None

    scored = []
    for key, values in numeric_fields:
        if not values:
            continue
        scored.append((key, values, _score_numeric_field(key, question_lower)))

    if not scored:
        return numeric_fields[0][0], numeric_fields[0][1]

    scored.sort(key=lambda item: item[2], reverse=True)
    best_key, best_values, best_score = scored[0]

    if best_score <= 0 and len(scored) > 1:
        # Prefer non-ID numeric columns if question doesn't clearly specify one.
        for key, values, score in scored:
            if not any(term in key.lower() for term in ["id", "code", "number"]):
                return key, values

    return best_key, best_values


def _describe_record(record):
    for label_key in ("name", "employee", "employee id", "id", "title", "person"):
        for key in record:
            if _normalize_key(key) == label_key:
                return record.get(key)
    # fallback to first non-empty string field
    for key, value in record.items():
        if isinstance(value, str) and value.strip():
            return value
    return "record"


def compute_analytics(question, vectorstore):
    data = extract_data(vectorstore)
    q = question.lower()

    if not data:
        return None

    if any(term in q for term in ["count", "total", "records"]):
        return f"Total records: {len(data)}"

    key, values = _pick_numeric_field(question, data)
    if values is None or key is None:
        return None

    if "average" in q or "mean" in q or "avg" in q:
        return f"Average {key} is {sum(values) / len(values):.2f}"

    if "median" in q:
        return f"Median {key} is {statistics.median(values)}"

    if "mode" in q:
        try:
            return f"Mode {key} is {statistics.mode(values)}"
        except statistics.StatisticsError:
            return "No unique mode found"

    if any(term in q for term in ["highest", "maximum", "max", "largest", "top"]):
        best = max(data, key=lambda x: _numeric_value(x.get(key)) or float("-inf"))
        name = _describe_record(best)
        return f"{name} has highest {key}: {best.get(key)}"

    if any(term in q for term in ["lowest", "minimum", "min", "smallest", "least"]):
        low = min(data, key=lambda x: _numeric_value(x.get(key)) or float("inf"))
        name = _describe_record(low)
        return f"{name} has lowest {key}: {low.get(key)}"

    if any(term in q for term in ["difference", "diff"]):
        return f"Difference between highest and lowest {key} is {max(values) - min(values)}"

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