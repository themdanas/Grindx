"""
retriever.py
------------
Builds and queries a FAISS index over embedded job chunks.
Also handles ATS cosine similarity scoring.
"""

import pickle
import numpy as np
import faiss

from embedder import embed_texts, get_embedding_client


def build_faiss_index(embeddings: list[list[float]]) -> tuple[faiss.Index, np.ndarray]:
    """
    Build a FAISS flat L2 index from a list of embedding vectors.

    Returns:
        (index, embedding_matrix) — the FAISS index and the numpy matrix.
    """
    embedding_matrix = np.array(embeddings).astype("float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dimension}).")
    return index, embedding_matrix


def save_index(index: faiss.Index, chunks: list[str], index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl"):
    """Persist the FAISS index and chunk list to disk."""
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index → {index_path}, chunks → {chunks_path}")


def load_index(index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl") -> tuple[faiss.Index, list[str]]:
    """Load a saved FAISS index and chunk list from disk."""
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded index ({index.ntotal} vectors) and {len(chunks)} chunks.")
    return index, chunks


def retrieve_jobs(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    client,
    top_k: int = 3,
) -> tuple[list[str], list[int]]:
    """
    Retrieve the top-k most relevant job chunks for a query.

    Returns:
        (top_chunks, indices) — matched text chunks and their positions.
    """
    query_emb = np.array(embed_texts([query], client)[0]).astype("float32").reshape(1, -1)
    distances, raw_indices = index.search(query_emb, top_k)
    indices = [int(i) for i in raw_indices[0]]
    top_chunks = [chunks[i] for i in indices]
    return top_chunks, indices


def compute_ats_score(
    resume_text: str,
    indices: list[int],
    embedding_matrix: np.ndarray,
    client,
) -> float:
    """
    Compute the ATS cosine similarity score between a resume and the
    top retrieved job chunk embeddings.

    Returns:
        Score as a percentage (0–100).
    """
    resume_emb = np.array(embed_texts([resume_text], client)[0]).astype("float32")

    scores = []
    for i in indices:
        chunk_emb = np.array(embedding_matrix[i])
        sim = np.dot(resume_emb, chunk_emb) / (
            np.linalg.norm(resume_emb) * np.linalg.norm(chunk_emb)
        )
        scores.append(sim)

    return float(max(scores) * 100)


def ats_pipeline(
    resume_text: str,
    index: faiss.Index,
    chunks: list[str],
    embedding_matrix: np.ndarray,
    client,
    top_k: int = 3,
) -> tuple[float, list[str]]:
    """
    Full ATS pipeline: retrieve relevant jobs and compute similarity score.

    Returns:
        (score, top_chunks)
    """
    top_chunks, indices = retrieve_jobs(resume_text, index, chunks, client, top_k)
    score = compute_ats_score(resume_text, indices, embedding_matrix, client)
    return score, top_chunks


if __name__ == "__main__":
    # Example: build index from scratch
    from parsar import load_dataframe, build_chunks
    from embedder import embed_texts, get_embedding_client

    client = get_embedding_client()
    df = load_dataframe("knowledge-based/*.csv")
    chunks = build_chunks(df)
    embeddings = embed_texts(chunks, client)

    index, embedding_matrix = build_faiss_index(embeddings)
    save_index(index, chunks)

    # Test retrieval
    results, idxs = retrieve_jobs("Python developer fresher", index, chunks, client)
    for r in results:
        print(r)
        print("=" * 50)