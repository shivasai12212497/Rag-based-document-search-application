import csv
import io
import json
import os
import re
import statistics
import subprocess
from functools import lru_cache

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# ================= CONFIG =================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ================= EMBEDDINGS =================
@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
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
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=_resolve_model(),
        base_url=OLLAMA_BASE_URL,
        temperature=0.2  # 🔥 faster + stable
    )

def cached_llm(prompt):
    return get_llm().invoke(prompt[:3500])

def _sanitize_answer(answer: str) -> str:
    if not answer:
        return ""
    lines = [line for line in answer.splitlines() if not line.strip().startswith("Role:")]
    return "\n".join(lines).strip()

# ================= QUERY EXPANSION =================
def expand_query(query: str):
    mapping = {
        "usa": ["new york", "san francisco"],
        "uk": ["london", "birmingham"],
        "uae": ["dubai"],
        "india": ["delhi", "mumbai", "bangalore", "hyderabad"]
    }

    query_lower = query.lower()
    expanded = [query]

    for key, values in mapping.items():
        if key in query_lower:
            expanded.extend(values)

    return " ".join(expanded)

# ================= DATA EXTRACTION =================
def parse_records(text: str):
    records = []
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        record = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip().lower()] = value.strip()
        if record:
            records.append(record)

    return records


FIELD_SYNONYMS = {
    "salary": ["salary", "pay", "income", "ctc"],
    "marks": ["marks", "score", "scores", "grades"],
    "age": ["age"],
    "year": ["year", "class"],
    "city": ["city", "location"],
    "course": ["course", "branch"],
    "department": ["department", "dept"],
    "job role": ["job role", "role", "position"],
}


def get_all_keys(records):
    keys = set()
    for record in records:
        keys.update(record.keys())
    return sorted(keys)


def to_number(value):
    try:
        return float(value)
    except Exception:
        return None


def detect_field(records, query):
    q = query.lower()

    for key in records[0].keys():
        if key.lower() in q:
            return key

    for field, words in FIELD_SYNONYMS.items():
        if any(word in q for word in words):
            for key in records[0].keys():
                if field in key.lower():
                    return key

    for key in records[0].keys():
        try:
            float(records[0].get(key))
            return key
        except Exception:
            continue

    return None


def extract_keywords(query):
    q = query.lower()
    stopwords = {
        "the", "is", "in", "of", "all", "with", "as", "having",
        "employee", "employees", "student", "students",
        "who", "which", "show", "list", "give", "me"
    }
    words = re.findall(r"\w+", q)
    return [w for w in words if w not in stopwords]


def smart_filter(records, query):
    stopwords = {"the", "is", "in", "of", "all", "with", "as", "having",
                 "employee", "employees", "student", "students",
                 "who", "which", "show", "list", "give", "me",
                 "how", "many", "number"}
    q_words = [word for word in query.lower().split() if len(word) > 2 and word not in stopwords]

    results = []
    for r in records:
        text = " ".join([str(v).lower() for v in r.values()])

        if all(word in text for word in q_words):
            results.append(r)
    return results


def count_records(records, query):
    q = query.lower()
    if not any(word in q for word in ["count", "how many", "number"]):
        return None

    filtered = smart_filter(records, query)
    if filtered:
        return f"I found {len(filtered)} matching records."

    return f"Total records: {len(records)}"


def compare_records(records, query):
    ids = re.findall(r"\d+", query)
    if len(ids) < 2:
        return None

    selected = []
    for r in records:
        for key in r:
            if "id" in key.lower() and str(r[key]) in ids:
                selected.append(r)
                break

    if len(selected) < 2:
        return None

    r1, r2 = selected[:2]
    response = "Here’s a clear comparison:\n\n"

    for key in r1.keys():
        if "id" in key.lower():
            continue

        v1 = r1.get(key)
        v2 = r2.get(key)
        try:
            n1 = float(v1)
            n2 = float(v2)
            if n1 > n2:
                better = r1.get("name")
            elif n2 > n1:
                better = r2.get("name")
            else:
                better = "Both are equal"
            response += f"{key}: {v1} vs {v2} → {better} performs better\n"
        except Exception:
            if str(v1).lower() == str(v2).lower():
                response += f"{key}: Both have {v1}\n"
            else:
                response += f"{key}: {v1} vs {v2}\n"
    return response


def get_name(record):
    return next((v for k, v in record.items() if "name" in k.lower()), "This record")


def get_max(records, query):
    field = detect_field(records, query)
    if not field:
        return "Please specify what you want the maximum of."

    valid = [r for r in records if to_number(r.get(field)) is not None]
    if not valid:
        return f"No numeric data found for {field}."

    best = max(valid, key=lambda x: to_number(x.get(field)))
    name = get_name(best)
    return f"Among all records, {name} has the highest {field} at {best[field]}."


def get_min(records, query):
    field = detect_field(records, query)
    if not field:
        return "Please specify what you want the minimum of."

    valid = [r for r in records if to_number(r.get(field)) is not None]
    if not valid:
        return f"No numeric data found for {field}."

    best = min(valid, key=lambda x: to_number(x.get(field)))
    name = get_name(best)
    return f"{name} has the lowest {field}, which is {best[field]}."


def get_avg(records, query):
    field = detect_field(records, query)
    if not field:
        return "Please tell me what you want the average of (e.g., marks, age)."

    values = []
    for r in records:
        try:
            val = float(r.get(field))
            values.append(val)
        except Exception:
            continue

    if not values:
        return f"I couldn't find numeric values for {field}."

    avg = sum(values) / len(values)
    return f"On average, the {field.lower()} across all records is about {round(avg, 2)}."


def get_record(records, query):
    q = query.lower()

    for r in records:
        for key in r:
            if "id" in key and str(r[key]) in q:
                return r

        name = r.get("name", "").lower()
        if name and any(part in q for part in name.split()):
            return r

    return None


def humanize_record(record):
    if not record:
        return "I couldn’t find that record."

    name = record.get("name", "This person")
    response = f"Here’s what I found about {name}:\n\n"

    for key, value in record.items():
        response += f"• {key.title()}: {value}\n"

    return response


def format_response(record, query):
    q = query.lower()
    if not record:
        return None

    if any(word in q for word in ["detail", "all", "profile"]):
        return humanize_record(record)

    for key in record:
        if key in q:
            if "role" in key.lower():
                return f"{record.get('name')} works as a {record[key]}."
            return f"{record.get('name')}'s {key} is {record[key]}."

    return humanize_record(record)


def format_list(records):
    if not records:
        return "I couldn’t find any matching records."

    response = f"I found {len(records)} matching records:\n\n"
    for r in records:
        name = r.get("name", "Unknown")
        role = r.get("job role", r.get("role", ""))
        dept = r.get("department", "")
        loc = r.get("location", r.get("city", ""))
        parts = [part for part in [role, dept, loc] if part]
        details = ", ".join(parts) if parts else "No additional details"
        response += f"• {name} ({details})\n"

    return response


def format_detailed_list(records):
    if not records:
        return "I couldn’t find any records."

    response = f"I found {len(records)} records:\n\n"
    for r in records:
        details = ", ".join([f"{k}: {v}" for k, v in r.items()])
        response += f"• {details}\n"
    return response


def structured_query_engine(records, query):
    q = query.lower()
    if not records:
        return "No data available."

    if any(word in q for word in ["who is", "about", "tell me"]):
        record = get_record(records, query)
        if record:
            return humanize_record(record)

    res = count_records(records, query)
    if res:
        return res

    record = get_record(records, query)
    if record:
        response = format_response(record, query)
        if response:
            return response

    if "average" in q:
        return get_avg(records, query)

    if "highest" in q:
        return get_max(records, query)

    if "lowest" in q:
        return get_min(records, query)

    filtered = smart_filter(records, query)
    if filtered:
        return format_list(filtered)

    return "I couldn’t find a clear match. Try rephrasing."


def detect_data_type(text):
    lines = text.split("\n")
    structured_count = sum(1 for line in lines if ":" in line)
    if structured_count > 5:
        return "structured"
    return "unstructured"


def is_structured_data(text):
    return "id" in text.lower() and ":" in text and detect_data_type(text) == "structured"


def text_search(text, query):
    query_words = query.lower().split()
    lines = text.split("\n")
    matches = []

    for line in lines:
        line_lower = line.lower()
        if any(word in line_lower for word in query_words):
            stripped = line.strip()
            if stripped:
                matches.append(stripped)
    return matches[:5]


def answer_from_text(text, query):
    matches = text_search(text, query)
    if not matches:
        return "I couldn’t find relevant information in the document."

    response = "Here’s what I found in the document:\n\n"
    for m in matches:
        response += f"• {m}\n"
    return response


def query_engine(text, query):
    data_type = detect_data_type(text)
    if data_type == "structured":
        records = parse_records(text)
        return structured_query_engine(records, query)
    return answer_from_text(text, query)


def split_documents(documents):
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    for doc in documents:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", doc.page_content.strip()) if block.strip()]
        if len(blocks) > 1 and all(":" in block for block in blocks):
            chunks.extend([Document(page_content=block, metadata=doc.metadata) for block in blocks])
            continue

        chunks.extend(splitter.split_documents([doc]))

    return chunks


def normalize_query(query):
    return query.lower().replace("air india", "").strip()


def retrieve_relevant_chunks(query, vectorstore, k=20, score_threshold=0.5):
    normalized_query = normalize_query(query)
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(normalized_query, k=k)
    except Exception:
        return vectorstore.similarity_search(normalized_query, k=min(k, 20))[:5]

    filtered_docs = []
    for doc, score in docs_and_scores:
        if score < score_threshold:
            filtered_docs.append(doc)
    return filtered_docs[:5]


def keyword_search(query, documents):
    query_words = [word for word in normalize_query(query).split() if word]
    matched = []

    for doc in documents:
        text = doc.page_content.lower()
        if any(word in text for word in query_words):
            matched.append(doc)

    return matched


def hybrid_search(query, vectorstore):
    all_docs = list(vectorstore.docstore._dict.values())
    vector_results = retrieve_relevant_chunks(query, vectorstore)
    keyword_results = keyword_search(query, all_docs)

    combined = []
    seen = set()
    for doc in vector_results + keyword_results:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            combined.append(doc)
            if len(combined) >= 5:
                break

    return combined

# ================= VECTORSTORE =================
def build_vectorstore_from_documents(documents):
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No valid content")
    return FAISS.from_documents(chunks, get_embeddings())

# ================= RAG =================
def rag_answer(question, vectorstore, chat_history=None):
    all_text = "\n\n".join([doc.page_content for doc in vectorstore.docstore._dict.values()])
    if is_structured_data(all_text):
        records = parse_records(all_text)
        answer = structured_query_engine(records, question)
        return {"result": answer, "source_documents": []}

    results = hybrid_search(question, vectorstore)
    print("Retrieved Chunks:")
    for doc in results:
        print(doc.page_content[:200])

    if not results:
        return {
            "result": "I checked the document, but couldn’t find a clear answer to that. Try asking in a simpler way or using keywords from the document.",
            "source_documents": []
        }

    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not clearly present, say you couldn’t find it.

Context:
{context}

Question:
{question}

Answer in a natural, human-friendly way.
"""

    answer = cached_llm(prompt)
    try:
        answer = remove_confidence(answer)
    except Exception:
        pass

    return {"result": answer.strip(), "source_documents": results}

# ================= CHAT =================
def chat_answer(question, chat_history=None):
    prompt = f"""
You are a fast assistant.
- Be concise
- No unnecessary explanations
- Do NOT include confidence score

User: {question}
Assistant:
"""
    answer = cached_llm(prompt)
    answer = remove_confidence(answer)
    return {"result": answer.strip(), "source_documents": []}