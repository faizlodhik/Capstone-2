# qualitative_agent.py
# Qualitative agent: semantic search over ChromaDB + Gemini for a grounded, cited answer.
# Flat layout: this file sits in the capstone folder next to ingest.py and data/.
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent  # the capstone folder
load_dotenv(PROJECT_ROOT / "APIKEY.env")        # APIKEY.env must contain GEMINI_API_KEY=...

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"  # lowercase data/, must match ingest.py
COLLECTION_NAME = "enterprise-docs"             # must match ingest.py
# Chunks farther away than this are discarded before they reach Gemini (lower = more similar).
# Tune it by running a few queries and comparing good vs. bad match distances.
MAX_DISTANCE = 1.2
NO_ANSWER = "I cannot find this information in the provided documents."
# Verify this ID against Google's current model list; override with GEMINI_MODEL in APIKEY.env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Loaded once at import time (slow to load, so not per query).
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is missing from APIKEY.env.")
        _client = genai.Client(api_key=api_key)
    return _client


# Rules live in the system instruction (stable); context + question go in the
# user content (changes every query). The "Source N" citation format must match
# what the validator searches for.
SYSTEM_INSTRUCTION = """You are an enterprise documentation assistant.

Rules:
1. Answer using ONLY the context provided. Do not use outside knowledge.
2. If the answer is not supported by the context, reply exactly:
   "I cannot find this information in the provided documents."
3. Cite every claim as [Source N], where N is the source number in the context.
4. If the context only partly answers the question, answer the supported part and say what is missing.
5. Do not invent facts, policies, owners, dates, or metrics."""


def retrieve(query: str, top_k: int = 5, max_distance: float = MAX_DISTANCE) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma.get_collection(COLLECTION_NAME)

    top_k = min(top_k, collection.count())
    embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "content": document,
            "source": metadata["source"],
            "chunk": metadata["chunk"],
            "distance": distance,  # lower = more similar
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
        if distance <= max_distance  # drop weak matches
    ]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source {i}: {c['source']}]\n{c['content']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def run(query: str, top_k: int = 5) -> dict:
    chunks = retrieve(query, top_k)

    # Nothing close enough: refuse without calling Gemini (no hallucination risk, zero tokens).
    if not chunks:
        return {
            "answer": NO_ANSWER,
            "chunks": [],
            "model": GEMINI_MODEL,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    prompt = build_prompt(query, chunks)

    response = _get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    usage = response.usage_metadata
    # `or 0` so the tokenomics logger never receives None.
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    # Thinking tokens are billed as output but reported separately, so add them in.
    output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + (
        getattr(usage, "thoughts_token_count", 0) or 0
    )

    return {
        "answer": response.text or "Gemini returned no answer (possibly blocked or truncated).",
        "chunks": chunks,
        "model": GEMINI_MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": getattr(usage, "total_token_count", None),
    }


if __name__ == "__main__":
    # Quick manual test: python qualitative_agent.py
    result = run("What is our company's security policy?")
    print(result["answer"])
    print("\nSources retrieved:")
    for i, c in enumerate(result["chunks"], 1):
        print(f"  [{i}] {c['source']} (chunk {c['chunk']}, distance {c['distance']:.3f})")
    print(f"\nTokens: in={result['input_tokens']} out={result['output_tokens']}")