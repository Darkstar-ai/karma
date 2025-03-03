import os
import uuid
import logging
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Configuration
BASE_PDF_PATH = os.getenv("BASE_PDF_PATH", "/Users/tfazio/Downloads/CODING/UTILITY/notebooks/RAG/docs/")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_PATH = os.getenv("QDRANT_PATH", ":memory:")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "collection_name")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "facebook/bart-large-cnn") # distilgpt2
LLM_MODEL_TASK = os.getenv("LLM_MODEL_TASK", "summarization") # text-generation

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QdrantDbManager:
    """Manage Qdrant vector DB operations."""

    def __init__(self, db_path: str, collection_name: str, embedding_model: HuggingFaceEmbeddings):
        self.db_instance = QdrantClient(path=db_path)
        self.create_collection(collection_name)
        self.collection_name = collection_name
        self.vector_store = QdrantVectorStore(
            client=self.db_instance, collection_name=collection_name, embedding=embedding_model, retrieval_mode=RetrievalMode.DENSE
        )

    def create_collection(self, collection_name: str):
        if not self.db_instance.collection_exists(collection_name=collection_name):
            self.db_instance.create_collection(
                collection_name=collection_name, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def add_documents(self, documents: List):
        if not documents:
            return
        logging.info(f"Adding {len(documents)} docs into {self.collection_name} collection...")
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata["chunk_id"])) for doc in documents]
        self.vector_store.add_documents(documents=documents, ids=ids)

    def search_docs(self, query: str, top_k: int = 5):
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)


def configure_environment():
    """Configure the environment and secrets."""
    # Load environment variables or secrets from a secure source
    pass


def load_pdf_into_list(base_pdf_path: str, file_name: str) -> List:
    """Load a PDF into a list of Documents."""
    try:
        file_path = os.path.join(base_pdf_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".pdf"):
            logging.info(f"Processing file: {file_name}")
            return PyPDFLoader(file_path).load()
        return []
    except Exception as e:
        logging.error(f"Error loading PDF {file_name}: {e}")
        return []


def split_document(doc_pages: List, chunk_size: int, chunk_overlap: int) -> List:
    """Recursive chunking strategy."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(doc_pages)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_{chunk.metadata['page']}_{chunk.metadata['start_index']}"
    return chunks


def setup_llm_pipeline(model_name: str):
    """Set up the language model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(LLM_MODEL_TASK, model=model, tokenizer=tokenizer, max_length=500)
    return HuggingFacePipeline(pipeline=pipe)


def main():
    """Main function to orchestrate the data processing and querying."""
    configure_environment()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    qdrant_manager = QdrantDbManager(QDRANT_PATH, COLLECTION_NAME, embedding_model)

    for file_name in os.listdir(BASE_PDF_PATH):
        file_pages = load_pdf_into_list(BASE_PDF_PATH, file_name)
        doc_splitted = split_document(file_pages, CHUNK_SIZE, CHUNK_OVERLAP)
        qdrant_manager.add_documents(doc_splitted)

    llm = setup_llm_pipeline(LLM_MODEL_NAME)
    rag_query = "deep learning volatility"
    llm_query = "Summarise the documents provided in the context."

    context_docs = qdrant_manager.search_docs(rag_query)
    context = "\n".join([f"Document {idx+1}: {context_docs[idx][0].page_content}" for idx in range(5)])
    prompt = (
        "You are a knowledgeable assistant. Based on the context provided below, please answer the query in detail.\n"
        "Context:\n"
        "{context}\n"
        "Query:\n"
        "{query}\n"
        "Provide a clear and concise answer."
    ).format(context=context, query=llm_query)
    result = llm(context, max_length=250, min_length=50, do_sample=False)
    logging.info(f"LLM Result: {result}")


if __name__ == "__main__":
    main()
