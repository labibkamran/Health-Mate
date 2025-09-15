import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def _pinecone_settings():
    return {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "index": os.getenv("PINECONE_INDEX", "health-mate-index"),
        "namespace": os.getenv("PINECONE_NAMESPACE", "default"),
        # For serverless indexes (recommended). Adjust if you use classic.
        "cloud": os.getenv("PINECONE_CLOUD", "aws"),
        "region": os.getenv("PINECONE_REGION", "us-east-1"),
    }

# text-embedding-004 returns 768-d vectors
EMBEDDING_DIM = 768


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def _get_embeddings():
    return VertexAIEmbeddings(model_name="text-embedding-004")


def _ensure_pinecone_index(index_name: str):
    settings = _pinecone_settings()
    if not settings["api_key"]:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=settings["api_key"])
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings["cloud"], region=settings["region"]),
        )


def _get_vectorstore(docs: List[Document] | None = None):
    settings = _pinecone_settings()
    _ensure_pinecone_index(settings["index"])
    embeddings = _get_embeddings()
    if docs:
        # Upsert documents to Pinecone
        return PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=settings["index"],
            namespace=settings["namespace"],
        )
    # Connect to existing Pinecone index
    return PineconeVectorStore(
        index_name=settings["index"],
        embedding=embeddings,
        namespace=settings["namespace"],
    )


def ingest_pdf(file_path: str) -> Tuple[int, int]:
    docs = load_pdf(file_path)
    chunks = chunk_docs(docs)
    _get_vectorstore(chunks)  # creates/updates index with these chunks
    return len(docs), len(chunks)


def retrieve_relevant(query: str, k: int = 4):
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search(query, k=k)
