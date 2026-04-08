"""
feedback.py
-----------
Generates ATS feedback using an Azure OpenAI LLM (gpt-4o-mini or similar).
"""

"""
feedback.py
-----------
Generates ATS resume feedback using Claude claude-sonnet-4-5
via Azure AI Foundry (AnthropicFoundry client).

Requirements:
    pip install anthropic[foundry] python-dotenv

.env variables needed:
    AZURE_FOUNDRY_RESOURCE_NAME=your-resource-name
    AZURE_SECRET_KEY=your-azure-api-key
"""

import re
import os
import dotenv
from anthropic import AnthropicFoundry


# ── Client ────────────────────────────────────────────────────────────────────

def get_llm_client() -> AnthropicFoundry:
    dotenv.load_dotenv("f:/grindx/.env")

    resource_name = os.getenv("AZURE_FOUNDRY_RESOURCE_NAME")
    api_key = os.getenv("AZURE_SECRET_KEY")

    if not resource_name:
        raise ValueError("Missing env var: AZURE_FOUNDRY_RESOURCE_NAME")
    if not api_key:
        raise ValueError("Missing env var: AZURE_SECRET_KEY")

    print("RESOURCE:", resource_name)

    client = AnthropicFoundry(
        base_url=f"https://{resource_name}.services.ai.azure.com/anthropic",
        api_key=api_key,
    )

    return client

# ── Keyword Extraction ────────────────────────────────────────────────────────

# Extend this list with any domain-specific tech your jobs require
TECH_KEYWORDS = [
    "Python", "SQL", "AWS", "GCP", "Azure", "Docker", "Kubernetes",
    "FastAPI", "Flask", "Django", "React", "Node", "TypeScript",
    "TensorFlow", "PyTorch", "scikit-learn", "Hugging Face", "LangChain",
    "LLM", "RAG", "NLP", "MLOps", "Airflow", "Spark", "Kafka",
    "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Git", "GitHub", "CI/CD", "REST", "GraphQL",
    "pandas", "NumPy", "matplotlib", "seaborn",
    "Linux", "Bash", "Terraform", "Ansible",
]

KEYWORD_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in TECH_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


def extract_keywords(text: str) -> set[str]:
    """Return a lowercase set of matched tech keywords from text."""
    matches = KEYWORD_PATTERN.findall(text)
    return {m.lower() for m in matches}


# ── Prompt Builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior technical recruiter with 10+ years of hiring experience
across product and service companies. You review resumes against job descriptions and give
honest, specific, actionable feedback. You never give generic advice — you always name
exact technologies, exact problems, and exact fixes."""


def build_prompt(
    resume: str,
    job_context: str,
    score: float,
    resume_keywords: set[str],
    missing_keywords: set[str],
) -> str:
    return f"""
You are reviewing a candidate's resume against the top matching job descriptions
retrieved from a job knowledge base.

══════════════════════════════════════
CANDIDATE RESUME
══════════════════════════════════════
{resume}

══════════════════════════════════════
TOP MATCHING JOB DESCRIPTIONS
══════════════════════════════════════
{job_context}

══════════════════════════════════════
ATS ANALYSIS SUMMARY
══════════════════════════════════════
ATS Match Score      : {score:.2f}%
Keywords in resume   : {', '.join(sorted(resume_keywords)) or 'none detected'}
Keywords MISSING     : {', '.join(sorted(missing_keywords)) or 'none — great coverage!'}

══════════════════════════════════════
YOUR TASK
══════════════════════════════════════
Give structured feedback under these exact sections:

1. HARD SKILLS GAP
   - List every technical skill or tool present in the job descriptions but absent from the resume.
   - Be specific: tool names, versions, frameworks.
   - If no gap, say "No hard skill gaps found."

2. SOFT SKILLS & EXPERIENCE GAP
   - Flag missing signals: leadership, teamwork, ownership, communication.
   - Note if seniority level (entry/mid/senior) seems mismatched.

3. RESUME FORMAT ISSUES
   - Identify vague bullet points with no metrics or impact.
   - Flag placeholder text (e.g. "Your College Name", "XX%").
   - Point out missing sections (e.g. no GitHub link, no project dates).

4. TOP 3 QUICK WINS
   - The 3 specific changes that will raise the ATS score the fastest.
   - Each quick win must be one concrete, actionable sentence.

5. OVERALL VERDICT
   - One of: STRONG HIRE / HIRE / MAYBE / NEEDS WORK / NOT A FIT
   - Follow with a single sentence explaining why.

Be direct. Be specific. Do not pad with encouragement.
"""


# ── Main Function ─────────────────────────────────────────────────────────────

def generate_feedback(
    resume: str,
    job_context: str,
    score: float,
    client: AnthropicFoundry,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 1500,
) -> str:
    """
    Generate structured ATS feedback for a resume.

    Args:
        resume      : Full resume text.
        job_context : Concatenated top retrieved job chunks from FAISS.
        score       : ATS cosine similarity score (0-100).
        client      : AnthropicFoundry client instance.
        model       : Must match your Azure Foundry deployment name exactly.
        max_tokens  : Max tokens in Claude's response.

    Returns:
        Formatted feedback string.
    """
    resume_keywords  = extract_keywords(resume)
    job_keywords     = extract_keywords(job_context)
    missing_keywords = job_keywords - resume_keywords

    prompt = build_prompt(resume, job_context, score, resume_keywords, missing_keywords)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.content[0].text


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = get_llm_client()

    sample_resume = """
    Name: Md Anas
    Skills: Python, pandas, NumPy, basic ML, RAG pipelines, Git
    Projects:
      - ATS Resume Analyzer: built using embeddings and FAISS retrieval
      - College Instagram Growth: grew page from 0 to active engagement
    Education: B.Tech Computer Science (in progress)
    """

    sample_job_context = """
    Data Scientist — requires Python, SQL, scikit-learn, PyTorch,
    A/B testing, statistical modeling, Spark, and cloud (AWS/GCP).

    AI Engineer — requires Python, LLM fine-tuning, Docker, Kubernetes,
    FastAPI, Hugging Face, MLOps, CI/CD pipelines.
    """

    score = 48.80

    print("Generating feedback via Azure Foundry...\n")
    feedback = generate_feedback(sample_resume, sample_job_context, score, client)
    print(feedback)