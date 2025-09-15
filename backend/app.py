import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment early so downstream imports see env vars (e.g., Pinecone key)
load_dotenv()

from src.config import init_vertex_ai, GCP_VERTEX_MODEL
from src.utils.rag import ingest_pdf, retrieve_relevant
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

vertex_info = init_vertex_ai()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "vertex": vertex_info}), 200


@app.post("/ingest")
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f.filename)
    f.save(file_path)

    num_docs, num_chunks = ingest_pdf(file_path)
    return jsonify({"ingested_documents": num_docs, "chunks": num_chunks}), 200


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Retrieve context
    context_docs = retrieve_relevant(query, k=4)
    context_text = "\n\n".join(d.page_content for d in context_docs)

    # LLM call
    llm = ChatVertexAI(model=GCP_VERTEX_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers using the provided context. If uncertain, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm

    answer = chain.invoke({"context": context_text, "question": query})
    return jsonify({
        "answer": answer.content,
        "sources": [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page", None)}
            for d in context_docs
        ],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
