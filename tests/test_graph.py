"""Tests for the graph topology and routing functions."""

from app.core.graph import after_examiner, should_continue, build_tutor_graph
from app.utils.pydantic_config import TutorState


def _make_state(**overrides) -> TutorState:
    """Helper to create a TutorState with sensible defaults."""
    defaults = {
        "session_id": "test-session",
        "topic": "Docker",
        "total_questions": 3,
        "current_question": 1,
        "status": "waiting_for_user",
    }
    defaults.update(overrides)
    return TutorState(**defaults)


class TestAfterExaminer:
    """Tests for the after_examiner routing function."""

    def test_routes_to_end_when_done(self):
        """Should return 'end' when status is 'done'."""
        state = _make_state(status="done")
        assert after_examiner(state) == "end"

    def test_routes_to_evaluate_when_waiting(self):
        """Should return 'evaluate' when status is not 'done'."""
        state = _make_state(status="waiting_for_user")
        assert after_examiner(state) == "evaluate"

    def test_routes_to_evaluate_when_evaluated(self):
        """Should return 'evaluate' for any non-done status."""
        state = _make_state(status="evaluated")
        assert after_examiner(state) == "evaluate"


class TestShouldContinue:
    """Tests for the should_continue routing function."""

    def test_routes_to_end_when_all_questions_answered(self):
        """Should return 'end' when current >= total."""
        state = _make_state(current_question=3, total_questions=3)
        assert should_continue(state) == "end"

    def test_routes_to_end_when_exceeded(self):
        """Should return 'end' when current > total (safety)."""
        state = _make_state(current_question=5, total_questions=3)
        assert should_continue(state) == "end"

    def test_routes_to_next_when_more_questions(self):
        """Should return 'next_question' when more questions remain."""
        state = _make_state(current_question=1, total_questions=3)
        assert should_continue(state) == "next_question"

    def test_routes_to_next_at_boundary(self):
        """Should return 'next_question' when exactly one question remains."""
        state = _make_state(current_question=2, total_questions=3)
        assert should_continue(state) == "next_question"


class TestBuildTutorGraph:
    """Tests for the graph builder function."""

    def test_graph_compiles_successfully(self):
        """The graph should compile without errors."""
        graph = build_tutor_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """The compiled graph should contain all 4 agent nodes."""
        graph = build_tutor_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "researcher" in node_names
        assert "summarizer" in node_names
        assert "examiner" in node_names
        assert "evaluator" in node_names
