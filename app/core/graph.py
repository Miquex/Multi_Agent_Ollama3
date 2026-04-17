"""LangGraph state machine definition for the Multi-Agent Tutor.

Defines the graph topology: researcher → summarizer → examiner → evaluator,
with conditional edges for the question loop and examiner guard.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.utils.pydantic_config import TutorState
from app.core.agents.tutor_agents import (
    researcher_node,
    summarizer_node,
    examiner_node,
    evaluator_node,
)
from app.utils.logger import logger


def after_examiner(state: TutorState) -> str:
    """Routes after the examiner node based on the guard status.

    If the examiner's hard guard triggered (status became ``"done"``),
    skips the evaluator entirely and routes to END.

    Args:
        state: The current tutor state after the examiner ran.

    Returns:
        ``"end"`` to skip evaluator, or ``"evaluate"`` to proceed normally.
    """
    if state.status == "done":
        logger.info("Examiner guard triggered — routing to END, skipping evaluator")
        return "end"
    return "evaluate"


def should_continue(state: TutorState) -> str:
    """Routes after the evaluator to either loop or finish.

    Compares ``current_question`` against ``total_questions`` to decide
    whether to generate the next question or end the session.

    Args:
        state: The current tutor state after the evaluator ran.

    Returns:
        ``"next_question"`` to loop back to examiner, or ``"end"`` to finish.
    """
    logger.info(
        f"should_continue check: current_question={state.current_question}, "
        f"total_questions={state.total_questions}"
    )
    if state.current_question >= state.total_questions:
        logger.info("Routing to END — all questions answered")
        return "end"
    logger.info("Routing to next_question")
    return "next_question"


def build_tutor_graph() -> StateGraph:
    """Constructs and compiles the tutor LangGraph state machine.

    Graph topology::

        START → researcher → summarizer → examiner
                                            ↓ (guard check)
                                         evaluator ←→ examiner (loop)
                                            ↓
                                           END

    The graph uses ``interrupt_before=["evaluator"]`` to pause execution
    and wait for the student's answer before grading.

    Returns:
        A compiled LangGraph application with memory checkpointing.
    """
    builder = StateGraph(TutorState)
    builder.add_node("researcher", researcher_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("examiner", examiner_node)
    builder.add_node("evaluator", evaluator_node)

    # Flow: researcher → summarizer → examiner → (check guard) → evaluator
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "summarizer")
    builder.add_edge("summarizer", "examiner")

    # After examiner: check if guard triggered, or proceed to evaluator
    builder.add_conditional_edges(
        "examiner",
        after_examiner,
        {"evaluate": "evaluator", "end": END},
    )

    # After evaluator: conditional routing (loop or end)
    builder.add_conditional_edges(
        "evaluator",
        should_continue,
        {"next_question": "examiner", "end": END},
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["evaluator"])


tutor_graph = build_tutor_graph()
