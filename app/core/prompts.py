"""Prompt templates for the Multi-Agent Tutor.

Centralizes all LLM prompt strings so they can be versioned,
reviewed, and modified without touching agent logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Summarizer Prompts ───────────────────────────────────────────────

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert tutor. Based ONLY on the provided context, "
            "create a clear and structured summary covering the key ideas, "
            "most important concepts, and essential takeaways about the topic. "
            "Use bullet points or numbered lists when appropriate.\n"
            "{format_instructions}",
        ),
        (
            "user",
            "Context: {context}\nTopic: {topic}\n\nProvide a comprehensive summary:",
        ),
    ]
)

SUMMARIZER_FALLBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert tutor. Based ONLY on the provided context, "
            "create a clear summary covering key ideas and important "
            "concepts. Use bullet points.",
        ),
        (
            "user",
            "Context: {context}\nTopic: {topic}\n\nProvide a summary:",
        ),
    ]
)

# ── Examiner Prompts ─────────────────────────────────────────────────

EXAMINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert tutor. You MUST ask ONE question that tests "
            "a key idea already explained in the Summary below. Do NOT ask "
            "about anything not covered in the Summary.\n"
            "Do NOT repeat any of the previous questions listed below.\n\n"
            "Summary:\n{summary}\n\n"
            "Previous questions asked:\n{previous_questions}\n\n"
            "{format_instructions}",
        ),
        (
            "user",
            "Topic: {topic}\n\nAsk a question about something from the summary:",
        ),
    ]
)

EXAMINER_FALLBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert tutor. Ask ONE short question that tests "
            "a key idea from the Summary below. Do NOT ask about anything "
            "not in the Summary.\n"
            "Do NOT repeat these previous questions:\n{previous_questions}\n"
            "Reply with ONLY the question, nothing else.\n\n"
            "Summary:\n{summary}",
        ),
        ("user", "Topic: {topic}"),
    ]
)

# ── Evaluator Prompts ────────────────────────────────────────────────

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict but fair tutor grading a student. "
            "Compare their answer to the true context.\n"
            "{format_instructions}",
        ),
        (
            "user",
            "Context: {context}\nQuestion Asked: {question}\n"
            "Student's Answer: {user_answer}\n\n"
            "Grade the answer and provide feedback",
        ),
    ]
)
