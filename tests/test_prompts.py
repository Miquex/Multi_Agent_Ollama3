"""Tests for the prompt templates module."""

from app.core.prompts import (
    SUMMARIZER_PROMPT,
    EXAMINER_PROMPT,
    EXAMINER_FALLBACK_PROMPT,
    EVALUATOR_PROMPT,
)


class TestPromptTemplates:
    """Tests to verify all prompt templates are well-formed."""

    def test_summarizer_prompt_has_required_variables(self):
        """Summarizer prompt should accept context and topic."""
        variables = SUMMARIZER_PROMPT.input_variables
        assert "context" in variables
        assert "topic" in variables

    def test_examiner_prompt_has_required_variables(self):
        """Examiner prompt should accept summary, topic, and previous_questions."""
        variables = EXAMINER_PROMPT.input_variables
        assert "summary" in variables
        assert "topic" in variables
        assert "previous_questions" in variables

    def test_examiner_fallback_has_required_variables(self):
        """Examiner fallback should accept summary, topic, and previous_questions."""
        variables = EXAMINER_FALLBACK_PROMPT.input_variables
        assert "summary" in variables
        assert "topic" in variables
        assert "previous_questions" in variables

    def test_evaluator_prompt_has_required_variables(self):
        """Evaluator prompt should accept summary, question, and user_answer."""
        variables = EVALUATOR_PROMPT.input_variables
        assert "summary" in variables
        assert "question" in variables
        assert "user_answer" in variables

    def test_all_prompts_are_chat_templates(self):
        """All prompts should be ChatPromptTemplate instances."""
        for prompt in [
            SUMMARIZER_PROMPT,
            EXAMINER_PROMPT,
            EXAMINER_FALLBACK_PROMPT,
            EVALUATOR_PROMPT,
        ]:
            assert hasattr(prompt, "format_messages"), (
                f"{prompt} is not a ChatPromptTemplate"
            )

    def test_prompt_count(self):
        """There should be exactly 4 prompt templates exported."""
        prompts = [
            SUMMARIZER_PROMPT,
            EXAMINER_PROMPT,
            EXAMINER_FALLBACK_PROMPT,
            EVALUATOR_PROMPT,
        ]
        assert len(prompts) == 4

