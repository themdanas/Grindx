"""
app.py
------
Gradio UI for the ATS Resume Analyzer.
Loads the pre-built FAISS index and chunks, then accepts resume uploads.

Usage:
    python app.py

Prerequisites:
    Run the retriever.py __main__ block once to build and save the index:
        python retriever.py
"""

import numpy as np
import gradio as gr
from PyPDF2 import PdfReader

from embedder import get_embedding_client
from retriever import load_index, ats_pipeline, build_faiss_index
from feedback import get_llm_client, generate_feedback


# ── Load clients and index once at startup ────────────────────────────────────

embedding_client = get_embedding_client()
llm_client = get_llm_client()

index, chunks = load_index("faiss_index.bin", "chunks.pkl")

# Rebuild embedding_matrix from saved index vectors (needed for ATS scoring)
embedding_matrix = np.zeros((index.ntotal, index.d), dtype="float32")
index.reconstruct_n(0, index.ntotal, embedding_matrix)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(file) -> str:
    """Extract plain text from an uploaded PDF or TXT file."""
    text = ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file.name)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def process_resume(file) -> tuple[str, str]:
    """Full pipeline: extract → score → retrieve → feedback."""
    resume_text = extract_text(file)

    score, top_chunks = ats_pipeline(
        resume_text, index, chunks, embedding_matrix, embedding_client, top_k=3
    )

    job_context = "\n\n".join(top_chunks)
    feedback = generate_feedback(resume_text, job_context, score, llm_client)

    return f"ATS Score: {score:.2f}%", feedback


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks() as app:
    gr.Markdown("# 📄 ATS Resume Analyzer")
    gr.Markdown("Upload your resume (PDF or TXT) to get an ATS score and actionable feedback.")

    file_input = gr.File(label="Upload Resume (.pdf or .txt)")
    score_output = gr.Textbox(label="ATS Score")
    feedback_output = gr.Textbox(label="Feedback", lines=15)

    btn = gr.Button("Analyze")
    btn.click(fn=process_resume, inputs=file_input, outputs=[score_output, feedback_output])

if __name__ == "__main__":
    app.launch(share=True)