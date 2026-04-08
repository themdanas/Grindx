import glob
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_dataframe(pattern: str) -> pd.DataFrame:
    """Load and combine all CSV files matching the given glob pattern."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {pattern}")
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} file(s).")
    return df


def row_to_text(row: pd.Series) -> str:
    """Convert a single DataFrame row into a structured text string."""
    return f"""
    Job Title: {row['Job Title']}
    Seniority Level: {row['Seniority Level']}
    Experience Required: {row['Experience Required']}

    Job Description:
    {row['Job Description']}
    """


def row_to_metadata(row: pd.Series) -> dict:
    """Extract metadata fields from a row."""
    return {
        "job_title": row["Job Title"],
        "seniority": row["Seniority Level"],
        "experience": row["Experience Required"],
    }


def build_chunks(df: pd.DataFrame, chunk_size: int = 100, chunk_overlap: int = 0) -> list[str]:
    """
    Split each row's text into chunks using RecursiveCharacterTextSplitter.
    Returns a flat list of clean, UTF-8-safe string chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for _, row in df.iterrows():
        row_text = row_to_text(row)
        chunks = splitter.split_text(row_text)
        all_chunks.extend(chunks)

    # Clean: keep non-empty strings only, ensure UTF-8 safety
    all_chunks = [
        chunk.encode("utf-8", "ignore").decode()
        for chunk in all_chunks
        if isinstance(chunk, str) and chunk.strip()
    ]

    print(f"Generated {len(all_chunks)} chunks.")
    return all_chunks


if __name__ == "__main__":
    # Quick smoke test
    df = load_dataframe("knowledge-based/*.csv")
    chunks = build_chunks(df)
    print("Sample chunk:", chunks[0])
