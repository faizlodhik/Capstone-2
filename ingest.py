# ingest.py
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def chunk_document(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap

    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]


def ingest(
    docs_path: str = "./data/documents",
    chroma_path: str = "./data/chroma",
    collection_name: str = "enterprise-docs",
) -> None:
    source_dir = Path(docs_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {source_dir}")

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    files = sorted(source_dir.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {source_dir}")
        return

    total_chunks = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipped empty file: {file_path.name}")
            continue

        chunks = chunk_document(text)
        embeddings = model.encode(
            chunks,
            show_progress_bar=False,
        ).tolist()

        collection.delete(where={"source": file_path.name})

        ids = [
            f"{file_path.name}-{index}"
            for index in range(len(chunks))
        ]
        metadatas = [
            {
                "source": file_path.name,
                "chunk": index,
                "file_type": "txt",
            }
            for index in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"Ingested {len(chunks)} chunks from {file_path.name}")

    print(f"Ingestion complete: {total_chunks} total chunks")


if __name__ == "__main__":
    ingest()
