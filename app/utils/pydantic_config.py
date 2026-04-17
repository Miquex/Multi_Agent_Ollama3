"""Pydantic models for the Multi-Agent Tutor state and structured LLM outputs."""

from pydantic import BaseModel, Field


class QuestionHistoryEntry(BaseModel):
    """A single question-answer pair with its evaluation result.

    Attributes:
        question: The question that was asked to the student.
        user_answer: The student's response to the question.
        grade: The numerical grade scored out of 10.
        feedback: Constructive feedback explaining the grade.
    """

    question: str
    user_answer: str
    grade: float
    feedback: str


class TutorState(BaseModel):
    """Shared state flowing through the LangGraph tutor pipeline.

    This model is the single source of truth for every node in the graph.
    Each node reads the fields it needs and returns a partial dict to update.

    Attributes:
        session_id: Unique session identifier for the current user interaction.
        topic: The subject the student wants to learn about.
        context: RAG-retrieved document content relevant to the topic.
        source_documents: File names of the documents used for RAG context.
        summary: Generated summary of the key ideas about the topic.
        question: The current question being asked to the student.
        user_answer: The student's most recent answer.
        feedback: Evaluator feedback for the most recent answer.
        grade: Numerical grade for the most recent answer (out of 10).
        status: Current state machine status controlling UI flow.
        total_questions: How many questions the user requested for this session.
        current_question: Which question number the session is currently on.
        questions_history: Accumulated list of all graded question-answer pairs.
    """

    session_id: str = Field(description="Unique session identifier")
    topic: str
    context: str = ""
    source_documents: list[str] = Field(
        default_factory=list,
        description="Raw document names for the UI to render",
    )
    summary: str = ""
    question: str = ""
    user_answer: str | None = None
    feedback: str = ""
    grade: float | None = None
    status: str = "starting"
    total_questions: int = 3
    current_question: int = 0
    questions_history: list[QuestionHistoryEntry] = Field(
        default_factory=list,
        description="History of all Q&A pairs with grades",
    )


class SummaryResult(BaseModel):
    """Structured output schema for the summarizer agent.

    Attributes:
        summary: A clear, structured summary covering key ideas about the topic.
    """

    summary: str = Field(
        description="A clear, structured summary covering the key ideas "
        "and most important concepts about the topic",
    )


class QuestionResult(BaseModel):
    """Structured output schema for the examiner agent.

    Attributes:
        question: One short, specific comprehension question.
    """

    question: str = Field(
        description="One short, specific question testing the student's comprehension",
    )


class EvaluationResult(BaseModel):
    """Structured output schema for the evaluator agent.

    Attributes:
        grade: The numerical grade scored out of 10.
        feedback: Constructive feedback explaining the grade and correcting mistakes.
    """

    grade: float = Field(description="The grade scored out of 10, e.g. 8.5")
    feedback: str = Field(
        description="Constructive feedback explaining the grade "
        "and correcting mistakes",
    )
