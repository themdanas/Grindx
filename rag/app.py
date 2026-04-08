"""
app.py
------
Gradio UI for the ATS Resume Analyzer.
Loads the pre-built FAISS index and chunks, then accepts resume uploads.
"""

import numpy as np
import gradio as gr
from PyPDF2 import PdfReader

from embedder import get_embedding_client
from retriever import load_index, ats_pipeline, build_faiss_index
from feedback import get_llm_client, generate_feedback
from roadmap import generate_roadmap


# ── Load clients and index once at startup ────────────────────────────────────

embedding_client = get_embedding_client()
llm_client = get_llm_client()

index, chunks = load_index("faiss_index.bin", "chunks.pkl")

embedding_matrix = np.zeros((index.ntotal, index.d), dtype="float32")
index.reconstruct_n(0, index.ntotal, embedding_matrix)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(file) -> str:
    text = ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file.name)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
    return text


# ── State to pass ATS result to roadmap tab ───────────────────────────────────

ats_state = {}

def process_resume(file) -> tuple[str, str]:
    """Full pipeline: extract → score → retrieve → feedback."""
    resume_text = extract_text(file)

    score, top_chunks = ats_pipeline(
        resume_text, index, chunks, embedding_matrix, embedding_client, top_k=3
    )

    job_context = "\n\n".join(top_chunks)
    feedback = generate_feedback(resume_text, job_context, score, llm_client)

    # Save to state so roadmap tab can use it
    ats_state["score"] = round(score, 2)
    ats_state["missing_keywords"] = []   # add if your feedback.py returns these
    ats_state["mistakes"] = []           # add if your feedback.py returns these

    return f"ATS Score: {score:.2f}%", feedback


def run_roadmap(role: str, days: int) -> dict:
    if not role.strip():
        return {"error": "Please enter a target role"}
    
    # Use real ATS data if resume was analyzed, else use empty defaults
    ats_result = ats_state if ats_state else {
        "score": 0,
        "missing_keywords": [],
        "mistakes": []
    }

    return generate_roadmap(ats_result, role, int(days))


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks() as app:
    gr.Markdown("# 📄 ATS Resume Analyzer & Placement Prep")

    with gr.Tab("ATS Scanner"):
        gr.Markdown("Upload your resume (PDF or TXT) to get an ATS score and actionable feedback.")

        file_input = gr.File(label="Upload Resume (.pdf or .txt)")
        score_output = gr.Textbox(label="ATS Score")
        feedback_output = gr.Textbox(label="Feedback", lines=15)

        btn = gr.Button("Analyze", variant="primary")
        btn.click(fn=process_resume, inputs=file_input, outputs=[score_output, feedback_output])

    with gr.Tab("Roadmap Generator"):
        gr.Markdown("### 🗺️ Get a personalized day-by-day prep roadmap")
        gr.Markdown("Tip: Run the ATS Scanner first — your resume data will be used automatically.")

        with gr.Row():
            role_input = gr.Textbox(label="Target Role", placeholder="e.g. SDE Intern, Data Analyst")
            days_input = gr.Number(label="Days Available", value=30, minimum=7, maximum=180)

        roadmap_btn = gr.Button("Generate Roadmap", variant="primary")
        roadmap_output = gr.JSON(label="Your Personalized Roadmap")

        roadmap_btn.click(fn=run_roadmap, inputs=[role_input, days_input], outputs=roadmap_output)

if __name__ == "__main__":
    app.launch(share=True)