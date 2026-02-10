import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore/faiss_index"

def ingest_text_files():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print("Ingestion complete")

if __name__ == "__main__":
    ingest_text_files()
