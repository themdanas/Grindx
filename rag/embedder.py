"""
embedder.py
-----------
Handles Azure OpenAI embedding calls.
Supports batched embedding for large chunk lists.
"""

import os
import dotenv
from openai import AzureOpenAI


def get_embedding_client() -> AzureOpenAI:
    """Initialize and return the Azure OpenAI client for embeddings."""
    dotenv.load_dotenv("/Users/amanyadav/IDTH/Grindx/.env")
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_FOUNDRY_TEXT_EMBEDDING_3_SMALL"),
        api_version="2025-04-01-preview",
        api_key=os.getenv("AZURE_SECRET_KEY"),
        default_headers={"api-key": os.getenv("AZURE_SECRET_KEY")},
        default_query={"api-version": "2025-04-01-preview"},
    )
    return client


def embed_texts(
    texts: list[str],
    client: AzureOpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 50,
) -> list[list[float]]:
    """
    Embed a list of texts using Azure OpenAI in batches.

    Args:
        texts: List of strings to embed.
        client: An AzureOpenAI client instance.
        model: Embedding model deployment name.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


if __name__ == "__main__":
    client = get_embedding_client()
    sample = ["Python developer with ML experience", "Data Scientist with SQL skills"]
    embeddings = embed_texts(sample, client)
    print(f"Got {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")