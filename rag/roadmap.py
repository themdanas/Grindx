"""
roadmap.py
----------
Generates a personalized placement prep roadmap using Claude claude-sonnet-4-5
via Azure AI Foundry (AnthropicFoundry client).
"""

import os
import re
import json
import dotenv
from anthropic import AnthropicFoundry


# ── Client ────────────────────────────────────────────────────────────────────

def get_roadmap_client() -> AnthropicFoundry:
    dotenv.load_dotenv("/Users/amanyadav/IDTH/Grindx/.env")

    resource_name = os.getenv("AZURE_FOUNDRY_RESOURCE_NAME")
    api_key = os.getenv("AZURE_SECRET_KEY")

    if not resource_name:
        raise ValueError("Missing env var: AZURE_FOUNDRY_RESOURCE_NAME")
    if not api_key:
        raise ValueError("Missing env var: AZURE_SECRET_KEY")

    return AnthropicFoundry(
        base_url=f"https://{resource_name}.services.ai.azure.com/anthropic",
        api_key=api_key,
    )


# ── Prompt Builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert placement coach who creates highly specific,
actionable preparation roadmaps for students targeting tech roles.
You always recommend real, freely available resources by name.
You respond ONLY in valid JSON. No explanation, no markdown, no preamble."""


def build_roadmap_prompt(
    ats_result: dict,
    target_role: str,
    timeframe_days: int,
) -> str:
    score = ats_result.get("score", "N/A")
    missing = ats_result.get("missing_keywords", [])
    mistakes = ats_result.get("mistakes", [])

    weeks = max(1, min(timeframe_days // 7, 4))  # cap at 4 weeks
    
    print("WEEKS:", weeks)
    return f"""
══════════════════════════════════════
CANDIDATE ATS ANALYSIS
══════════════════════════════════════
ATS Score        : {score}%
Missing Keywords : {', '.join(sorted(missing)) if missing else 'Not provided'}
Resume Mistakes  : {', '.join(mistakes) if mistakes else 'Not provided'}

══════════════════════════════════════
ROADMAP REQUEST
══════════════════════════════════════
Target Role      : {target_role}
Timeframe        : {timeframe_days} days (~{weeks} weeks)

══════════════════════════════════════
YOUR TASK
══════════════════════════════════════
Generate a complete placement prep roadmap for this candidate.

Rules:
1. Create exactly {weeks} weeks of content
2. Weight topics by what's missing from the resume
3. Last week must be: Mock Interviews + Revision only
4. Resources must be real and free (YouTube channels, docs, LeetCode, GeeksForGeeks etc.)
5. Respond ONLY with this exact JSON structure, nothing else:

{{
  "target_role": "{target_role}",
  "total_days": {timeframe_days},
  "skill_gaps": ["gap1", "gap2", "gap3", "gap4", "gap5"],
  "roadmap": [
    {{
      "week": 1,
      "focus": "topic name",
      "daily_hours": 2,
      "topics": ["topic1", "topic2", "topic3"],
      "resources": [
        {{
          "title": "resource name",
          "type": "youtube/article/practice",
          "url": "https://..."
        }}
      ]
    }}
  ]
}}
"""


# ── Main Function ─────────────────────────────────────────────────────────────

def generate_roadmap(
    ats_result: dict,
    target_role: str,
    timeframe_days: int,
    client: AnthropicFoundry = None,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 1500,
) -> dict:
    """
    Generate a personalized placement prep roadmap.

    Args:
        ats_result      : Dict with keys: score, missing_keywords, mistakes
        target_role     : e.g. "SDE Intern", "Data Analyst"
        timeframe_days  : Number of days the candidate has to prepare
        client          : AnthropicFoundry client (creates one if not passed)
        model           : Must match your Azure Foundry deployment name exactly
        max_tokens      : Max tokens in Claude's response

    Returns:
        Parsed roadmap as a Python dict
    """
    if client is None:
        client = get_roadmap_client()

    prompt = build_roadmap_prompt(ats_result, target_role, timeframe_days)

    for attempt in range(3):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        text = response.content[0].text

        # Strip markdown fences if Claude wraps in ```json ... ```
        text = re.sub(r"```json|```", "", text).strip()

        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            # Try extracting outermost { }
            start = text.index('{')
            end = text.rindex('}') + 1
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                return {
                    "error": "Failed to parse roadmap after 3 attempts",
                    "raw_output": text[:500]
                }
            continue


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = get_roadmap_client()

    sample_ats = {
        "score": 52.4,
        "missing_keywords": ["Docker", "SQL", "System Design", "REST", "Kubernetes"],
        "mistakes": ["No quantified achievements", "Missing GitHub link", "Vague project descriptions"]
    }

    print("Generating roadmap via Azure Foundry...\n")
    roadmap = generate_roadmap(
        ats_result=sample_ats,
        target_role="SDE Intern",
        timeframe_days=30,
        client=client
    )

    import pprint
    pprint.pprint(roadmap)