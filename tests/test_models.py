"""Tests for Pydantic models in pydantic_config.py."""

import pytest
from pydantic import ValidationError

from app.utils.pydantic_config import (
    TutorState,
    QuestionHistoryEntry,
    SummaryResult,
    QuestionResult,
    EvaluationResult,
)


class TestQuestionHistoryEntry:
    """Tests for the QuestionHistoryEntry model."""

    def test_valid_entry(self):
        """A valid entry should be created without errors."""
        # Arrange & Act
        entry = QuestionHistoryEntry(
            question="What is Docker?",
            user_answer="A containerization platform",
            grade=8.5,
            feedback="Good answer!",
        )

        # Assert
        assert entry.question == "What is Docker?"
        assert entry.grade == 8.5

    def test_missing_required_field_raises(self):
        """Omitting a required field should raise ValidationError."""
        with pytest.raises(ValidationError):
            QuestionHistoryEntry(
                question="What is Docker?",
                # user_answer is missing
                grade=8.5,
                feedback="Good answer!",
            )


class TestTutorState:
    """Tests for the TutorState model."""

    def test_defaults(self):
        """Default values should be set correctly for a minimal state."""
        # Arrange & Act
        state = TutorState(session_id="abc-123", topic="Docker")

        # Assert
        assert state.context == ""
        assert state.status == "starting"
        assert state.total_questions == 3
        assert state.current_question == 0
        assert state.questions_history == []
        assert state.source_documents == []
        assert state.user_answer is None
        assert state.grade is None

    def test_full_state(self):
        """A fully populated state should work correctly."""
        # Arrange
        history_entry = QuestionHistoryEntry(
            question="Q1?",
            user_answer="A1",
            grade=7.0,
            feedback="OK",
        )

        # Act
        state = TutorState(
            session_id="test-session",
            topic="Python",
            context="Some context",
            summary="Key ideas...",
            question="What is Python?",
            user_answer="A programming language",
            status="waiting_for_user",
            total_questions=5,
            current_question=2,
            questions_history=[history_entry],
        )

        # Assert
        assert state.topic == "Python"
        assert state.current_question == 2
        assert len(state.questions_history) == 1
        assert state.questions_history[0].grade == 7.0

    def test_missing_session_id_raises(self):
        """session_id is required and should raise if missing."""
        with pytest.raises(ValidationError):
            TutorState(topic="Docker")

    def test_missing_topic_raises(self):
        """topic is required and should raise if missing."""
        with pytest.raises(ValidationError):
            TutorState(session_id="abc")


class TestSummaryResult:
    """Tests for the SummaryResult model."""

    def test_valid_summary(self):
        """A valid summary should parse correctly."""
        result = SummaryResult(summary="Docker is a containerization tool.")
        assert "Docker" in result.summary

    def test_empty_summary_allowed(self):
        """An empty string should be accepted (no min length)."""
        result = SummaryResult(summary="")
        assert result.summary == ""

    def test_missing_summary_raises(self):
        """Omitting summary should raise ValidationError."""
        with pytest.raises(ValidationError):
            SummaryResult()


class TestQuestionResult:
    """Tests for the QuestionResult model."""

    def test_valid_question(self):
        """A valid question should parse correctly."""
        result = QuestionResult(question="What is Docker?")
        assert result.question == "What is Docker?"

    def test_missing_question_raises(self):
        """Omitting question should raise ValidationError."""
        with pytest.raises(ValidationError):
            QuestionResult()


class TestEvaluationResult:
    """Tests for the EvaluationResult model."""

    def test_valid_evaluation(self):
        """A valid evaluation should parse correctly."""
        result = EvaluationResult(grade=8.5, feedback="Great answer!")
        assert result.grade == 8.5
        assert result.feedback == "Great answer!"

    def test_missing_grade_raises(self):
        """Omitting grade should raise ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationResult(feedback="Good")

    def test_missing_feedback_raises(self):
        """Omitting feedback should raise ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationResult(grade=5.0)
