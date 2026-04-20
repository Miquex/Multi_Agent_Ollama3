"""Tutor agent nodes for the LangGraph state machine.

Each function in this module is a LangGraph node that receives the shared
``TutorState``, performs its task (RAG retrieval, summarization, question
generation, or answer evaluation), and returns a partial state update dict.
"""

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from app.core.llm_factory import get_llm
from app.core.prompts import (
    SUMMARIZER_PROMPT,
    EXAMINER_PROMPT,
    EXAMINER_FALLBACK_PROMPT,
    EVALUATOR_PROMPT,
    QUESTION_FORMAT,
    EVALUATION_FORMAT,
)
from app.core.rag.vector_store_data import get_vector_store_manager
from app.utils.logger import logger
from app.utils.sanitizer import sanitize_user_input
from app.utils.pydantic_config import (
    TutorState,
    QuestionResult,
    EvaluationResult,
    QuestionHistoryEntry,
)


def researcher_node(state: TutorState) -> dict[str, object]:
    """Retrieves relevant documents from the vector store for the given topic.

    Uses the singleton ``VectorStoreManager`` to avoid re-initializing
    the embedding model on every call.

    Args:
        state: The current tutor state containing the topic to research.

    Returns:
        A dict with ``context`` (joined document text) and
        ``source_documents`` (list of file names).
    """
    logger.info(f"Agent 1 (Researcher) searching RAG for: {state.topic}")
    vms = get_vector_store_manager()
    retriever = vms.get_retriever()
    docs = retriever.invoke(state.topic)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source", "Unknown Document") for doc in docs]
    return {"context": context, "source_documents": sources}


def summarizer_node(state: TutorState) -> dict[str, str]:
    """Generates a structured summary of the retrieved context.

    Uses the LLM with a ``StrOutputParser`` to produce plain-text
    markdown directly, avoiding JSON parsing failures that cause
    costly fallback re-invocations.

    Args:
        state: The current tutor state containing context and topic.

    Returns:
        A dict with a ``summary`` string of key ideas and concepts.
    """
    logger.info(f"Agent (Summarizer) generating summary for: {state.topic}")
    llm = get_llm()
    chain = SUMMARIZER_PROMPT | llm | StrOutputParser()
    summary: str = chain.invoke(
        {
            "context": state.context,
            "topic": state.topic,
        }
    )
    logger.success(f"Summarizer generated summary ({len(summary)} chars)")
    return {"summary": summary.strip()}


def examiner_node(state: TutorState) -> dict[str, object]:
    """Generates a single comprehension question based on the summary.

    Includes a hard guard to prevent exceeding ``total_questions``.
    Questions are derived from the summary so they align with what
    the student has already read. Previous questions are excluded
    to avoid repetition.

    Args:
        state: The current tutor state containing summary, topic,
            question history, and question counters.

    Returns:
        A dict updating ``question``, ``current_question``, and ``status``.
        Returns ``{"status": "done"}`` if the guard triggers.
    """
    # Hard guard: never exceed total_questions
    if state.current_question >= state.total_questions:
        logger.warning(
            f"Examiner guard triggered: {state.current_question} >= "
            f"{state.total_questions}. Skipping question generation."
        )
        return {"status": "done"}

    new_question_num = state.current_question + 1
    logger.info(
        f"Agent 2 (Examiner) generating question {new_question_num} "
        f"of {state.total_questions}..."
    )
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=QuestionResult)

    # Build list of previous questions to avoid repetition
    previous_questions = (
        "\n".join([f"- {q.question}" for q in state.questions_history])
        if state.questions_history
        else "None"
    )

    chain = EXAMINER_PROMPT | llm | parser
    try:
        result: QuestionResult = chain.invoke(
            {
                "summary": state.summary,
                "topic": state.topic,
                "previous_questions": previous_questions,
                "format_instructions": QUESTION_FORMAT,
            }
        )
        logger.success(
            f"Examiner generated question {new_question_num}: {result.question}"
        )
        return {
            "question": result.question,
            "current_question": new_question_num,
            "status": "waiting_for_user",
        }
    except Exception as e:
        logger.error(f"Examiner failed structured output: {e}")
        fallback_chain = EXAMINER_FALLBACK_PROMPT | llm
        raw = fallback_chain.invoke(
            {
                "summary": state.summary,
                "topic": state.topic,
                "previous_questions": previous_questions,
            }
        )
        question_text = str(raw.content).strip()
        logger.info(f"Examiner fallback question {new_question_num}: {question_text}")
        return {
            "question": question_text,
            "current_question": new_question_num,
            "status": "waiting_for_user",
        }


def evaluator_node(state: TutorState) -> dict[str, object]:
    """Grades the student's answer and provides constructive feedback.

    Sanitizes the student's answer before evaluation to guard against
    prompt injection. Compares the answer against the RAG context,
    assigns a numerical grade, and appends the result to history.

    Args:
        state: The current tutor state containing the question,
            user answer, context, and question counters.

    Returns:
        A dict updating ``feedback``, ``grade``, ``questions_history``,
        and ``status`` (either ``"evaluated"`` or ``"done"``).
    """
    logger.info(
        f"Agent 3 (Evaluator) checking answer for question "
        f"{state.current_question} of {state.total_questions}..."
    )
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=EvaluationResult)
    chain = EVALUATOR_PROMPT | llm | parser

    # Sanitize user answer before sending to LLM
    clean_answer = sanitize_user_input(state.user_answer or "")

    try:
        result: EvaluationResult = chain.invoke(
            {
                "summary": state.summary,
                "question": state.question,
                "user_answer": clean_answer,
                "format_instructions": EVALUATION_FORMAT,
            }
        )
        grade = result.grade
        feedback = result.feedback
        logger.success(
            f"Evaluator graded question {state.current_question}: {grade}/10"
        )
    except Exception as e:
        logger.error(f"Evaluator failed structured output: {e}")
        grade = 0.0
        feedback = "The tutor had trouble grading. Please try again."

    # Append current Q&A to history
    updated_history = list(state.questions_history)
    updated_history.append(
        QuestionHistoryEntry(
            question=state.question,
            user_answer=state.user_answer or "",
            grade=grade,
            feedback=feedback,
        )
    )

    # Determine if we are done
    is_done = state.current_question >= state.total_questions
    new_status = "done" if is_done else "evaluated"

    logger.info(
        f"Question {state.current_question}/{state.total_questions} "
        f"— status: {new_status}"
    )

    return {
        "feedback": feedback,
        "grade": grade,
        "questions_history": updated_history,
        "status": new_status,
    }
