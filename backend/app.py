import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
from src.config import init_vertex_ai, GCP_VERTEX_MODEL
from vertexai.preview import rag as vertex_rag
from vertexai.preview.generative_models import GenerativeModel

app = Flask(__name__)

vertex_info = init_vertex_ai()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "vertex": vertex_info}), 200


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    query = data.get("query")
    debug = bool(data.get("debug"))
    if not query:
        return jsonify({"error": "Missing query"}), 400
    corpus_name = os.getenv("RAG_CORPUS")
    if not corpus_name:
        return jsonify({"error": "RAG_CORPUS not configured", "hint": "Set RAG_CORPUS in .env"}), 500

    try:
        model = GenerativeModel(GCP_VERTEX_MODEL)
        # Retrieve contexts directly (Managed RAG Retrieve API)
        retrieve_fn = getattr(vertex_rag, "retrieve", None)
        contexts_joined = ""
        citations = []
        debug_chunks = []
        if retrieve_fn:
            results = retrieve_fn(
                query,
                rag_resources=[vertex_rag.RagResource(rag_corpus=corpus_name)],
                top_k=4,
            )
            context_texts = []
            for r in getattr(results, "results", []) or []:
                for chunk in getattr(r, "contexts", []) or []:
                    text = getattr(chunk, "content", None) or getattr(chunk, "text", "")
                    if text:
                        context_texts.append(text)
                        if debug:
                            debug_chunks.append({
                                "preview": text[:200],
                                "len": len(text),
                                "uri": getattr(chunk, "uri", None),
                                "page": getattr(chunk, "page", None),
                            })
                    citations.append({
                        "uri": getattr(chunk, "uri", None),
                        "page": getattr(chunk, "page", None),
                    })
            contexts_joined = "\n\n".join(context_texts)[:12000]
            if debug:
                print(f"[RAG] Retrieved {len(context_texts)} contexts; joined length={len(contexts_joined)}")
        prompt = (
            "You are a friendly but professional AI medical assistant. "
            "Your knowledge comes primarily from authoritative sources provided as context "
            "(e.g., The Gale Encyclopedia of Medicine, Merck Manual, Mayo Clinic). "
            "Always base your answers on the given context. "
            "If the context is empty or does not contain relevant information, say politely that you don't know "
            "and suggest the user consult a qualified healthcare professional.\n\n"

            "Guidelines:\n"
            "- Provide clear, structured, and accurate medical explanations.\n"
            "- Use simple English or Urdu if the question suggests it; adapt to the user's language (English/Urdu/Roman Urdu).\n"
            "- If medical terminology is used, also provide a plain-language explanation for laypersons.\n"
            "- When the context is limited, avoid guessing or fabricating medical advice.\n"
            "- Never provide prescriptions, dosages, or medical treatment plans unless explicitly present in the context.\n"
            "- Always encourage professional medical consultation for diagnosis or urgent care.\n"
            "- If helpful, organize answers with bullet points, headings, or short summaries.\n"
            "- Be concise but informative.\n\n"

            f"Context:\n{contexts_joined}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = model.generate_content(prompt)
        resp = {
            "answer": getattr(response, "text", str(response)),
            "citations": citations,
        }
        if debug:
            resp["chunks"] = debug_chunks
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/debug/retrieve")
def debug_retrieve():
    data = request.get_json(force=True)
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing query"}), 400
    corpus_name = os.getenv("RAG_CORPUS")
    if not corpus_name:
        return jsonify({"error": "RAG_CORPUS not configured", "hint": "Set RAG_CORPUS in .env"}), 500

    retrieve_fn = getattr(vertex_rag, "retrieve", None)
    if not retrieve_fn:
        return jsonify({"error": "vertexai.preview.rag.retrieve not available in this SDK"}), 500

    try:
        results = retrieve_fn(
            query,
            rag_resources=[vertex_rag.RagResource(rag_corpus=corpus_name)],
            top_k=8,
        )
        payload = {
            "query": query,
            "corpus": corpus_name,
            "result_sets": []
        }
        for r in getattr(results, "results", []) or []:
            entry = {"contexts": []}
            for chunk in getattr(r, "contexts", []) or []:
                text = getattr(chunk, "content", None) or getattr(chunk, "text", "")
                entry["contexts"].append({
                    "preview": text[:300],
                    "len": len(text) if text else 0,
                    "uri": getattr(chunk, "uri", None),
                    "page": getattr(chunk, "page", None),
                })
            payload["result_sets"].append(entry)
        payload["total_contexts"] = sum(len(rs["contexts"]) for rs in payload["result_sets"])
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
