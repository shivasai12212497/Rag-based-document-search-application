from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline

VECTORSTORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
DEFAULT_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.25
MAX_NEW_TOKENS = 200


def load_rag_pipeline():
    # -------- Embeddings --------
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # -------- Load Vector Store --------
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # -------- LLM Pipeline --------
    hf_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        device=-1,  # CPU
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # -------- Helpers --------
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

    # -------- RAG Answer Function --------
    def rag_answer(question: str):
        question = (question or "").strip()
        if not question:
            return {
                "result": "Please enter a question.",
                "source_documents": [],
            }

        scored = vectorstore.similarity_search_with_relevance_scores(
            question, k=DEFAULT_TOP_K
        )
        docs = [doc for doc, score in scored if score >= MIN_RELEVANCE_SCORE]

        if not docs:
            return {
                "result": "I couldn't find relevant information in the documents.",
                "source_documents": [],
            }

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = build_prompt(question, context)
        answer = llm.invoke(prompt)

        return {
            "result": answer,
            "source_documents": docs,
        }

    return rag_answer
