import os
from functools import lru_cache
from typing import Sequence

# Force PyTorch-only Transformers usage to avoid TensorFlow/Keras runtime issues
# on environments that have Keras 3 but not tf-keras.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from transformers import pipeline

VECTORSTORE_PATH = "vectorstore/faiss_index" #ASTRADB
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL = "google/flan-t5-small" #GROQAI
DEFAULT_TOP_K = 5
MIN_COSINE_SIMILARITY = 0.35
MAX_NEW_TOKENS = 200
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DISTANCE_STRATEGY = DistanceStrategy.COSINE
NORMALIZE_L2 = True


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        device=-1,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


def split_documents(documents: Sequence[Document]):
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(list(documents))


def build_vectorstore_from_documents(documents: Sequence[Document]):
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No valid content found in uploaded files.")
    embeddings = get_embeddings()
    return FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DISTANCE_STRATEGY,
        normalize_L2=NORMALIZE_L2,
    )


def load_vectorstore(vectorstore_path: str = VECTORSTORE_PATH):
    embeddings = get_embeddings()
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DISTANCE_STRATEGY,
        normalize_L2=NORMALIZE_L2,
    )


def save_vectorstore(vectorstore, vectorstore_path: str = VECTORSTORE_PATH):
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    vectorstore.save_local(vectorstore_path)


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a careful assistant.\n"
        "Answer ONLY using the context.\n"
        "If the answer is not in the context, reply: "
        "\"I don't have information about that.\"\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _cosine_similarity_from_l2_distance(distance: float) -> float:
    # With normalized vectors, cosine similarity = 1 - (squared_l2_distance / 2)
    similarity = 1.0 - (distance / 2.0)
    return max(-1.0, min(1.0, similarity))


def rag_answer(question: str, vectorstore, top_k: int = DEFAULT_TOP_K):
    question = (question or "").strip()
    if not question:
        return {
            "result": "Please enter a question.",
            "source_documents": [],
        }

    scored = vectorstore.similarity_search_with_score(question, k=top_k)
    docs = [
        doc
        for doc, distance in scored
        if _cosine_similarity_from_l2_distance(distance) >= MIN_COSINE_SIMILARITY
    ]

    if not docs:
        return {
            "result": "I couldn't find relevant information in the uploaded documents.",
            "source_documents": [],
        }

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(question, context)
    answer = get_llm().invoke(prompt)

    return {
        "result": answer,
        "source_documents": docs,
    }


def load_rag_pipeline(vectorstore_path: str = VECTORSTORE_PATH):
    vectorstore = load_vectorstore(vectorstore_path)

    def answer(question: str):
        return rag_answer(question, vectorstore)

    return answer
