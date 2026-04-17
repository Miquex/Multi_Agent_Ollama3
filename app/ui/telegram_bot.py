"""Telegram bot interface for the Multi-Agent Tutor.

Provides a conversational Telegram bot that reuses the same LangGraph
tutor pipeline as the Streamlit UI. Supports:
- Learning sessions with summary, questions, and graded feedback
- YouTube transcript downloads to the knowledge base
- Per-user session management via chat_id-based thread IDs

Commands:
    /start   — Begin a new learning session
    /youtube — Download a YouTube transcript to the knowledge base
    /cancel  — Cancel the current operation
"""

import os
import uuid

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from langchain_core.runnables import RunnableConfig
from app.core.graph import tutor_graph
from app.utils.logger import logger
from app.utils.sanitizer import sanitize_topic, sanitize_user_input
from app.utils.youtube_fetcher import download_youtube_to_knowledge
from app.core.rag.vector_store_data import get_vector_store_manager

load_dotenv()

# Telegram has a 4096 character message limit
MAX_MESSAGE_LENGTH = 4096

# ── Conversation states ──────────────────────────────────────────────
TOPIC, QUESTION_COUNT, ANSWERING = range(3)
YOUTUBE_URL, YOUTUBE_TITLE = range(10, 12)


def _split_message(text: str) -> list[str]:
    """Splits a long message into Telegram-safe chunks.

    Args:
        text: The message text to split.

    Returns:
        A list of strings, each within the Telegram character limit.
    """
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        # Find a good split point (newline or space)
        split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = text.rfind(" ", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = MAX_MESSAGE_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


# ── Learning Session Handlers ────────────────────────────────────────


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles /start command — begins a new learning session."""
    assert update.message and update.effective_user
    logger.info(f"Telegram /start from user {update.effective_user.id}")
    await update.message.reply_text(
        "🎓 *Welcome to the Multi-Agent Tutor!*\n\n"
        "I'll research a topic, give you a summary, then quiz you.\n\n"
        "📝 *What would you like to learn today?*\n"
        "_(e.g. Docker, Python, Machine Learning)_",
        parse_mode="Markdown",
    )
    return TOPIC


async def receive_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the learning topic and asks for question count."""
    assert update.message and update.message.text and update.effective_user and context.user_data is not None
    topic = sanitize_topic(update.message.text)
    context.user_data["topic"] = topic
    logger.info(f"Telegram user {update.effective_user.id} chose topic: {topic}")

    await update.message.reply_text(
        f"Great choice! Topic: *{topic}*\n\n"
        "🔢 *How many questions would you like?* (1-10)",
        parse_mode="Markdown",
    )
    return QUESTION_COUNT


async def receive_question_count(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Receives question count, invokes the graph, and sends the first question."""
    assert update.message and update.message.text and update.effective_user and context.user_data is not None
    text = update.message.text.strip()

    # Validate number
    try:
        num_questions = int(text)
        if not 1 <= num_questions <= 10:
            raise ValueError
    except ValueError:
        await update.message.reply_text("⚠️ Please send a number between 1 and 10.")
        return QUESTION_COUNT

    # Create a fresh thread for this session
    thread_id = f"tg-{update.effective_user.id}-{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    context.user_data["config"] = config
    context.user_data["num_questions"] = num_questions

    await update.message.reply_text(
        "🔍 Agents are researching and preparing your session...\n"
        "This may take a moment ⏳"
    )

    # Invoke the graph (runs researcher → summarizer → examiner, then pauses)
    topic = context.user_data["topic"]
    try:
        state = tutor_graph.invoke(
            {
                "topic": topic,
                "session_id": thread_id,
                "total_questions": num_questions,
                "current_question": 0,
                "questions_history": [],
            },
            config,
        )
        context.user_data["graph_state"] = state
    except Exception as e:
        logger.error(f"Graph invoke failed for Telegram user: {e}")
        await update.message.reply_text(
            "❌ Something went wrong. Please try again with /start"
        )
        return ConversationHandler.END

    # Send the summary
    summary = state.get("summary", "No summary available.")
    for chunk in _split_message(f"📝 *Topic Summary*\n\n{summary}"):
        await update.message.reply_text(chunk, parse_mode="Markdown")

    # Send the first question
    question = state.get("question", "")
    current_q = state.get("current_question", 1)
    total_q = state.get("total_questions", 1)

    await update.message.reply_text(
        f"📚 *Question {current_q} of {total_q}*\n\n"
        f"{question}\n\n"
        "_Type your answer below:_",
        parse_mode="Markdown",
    )
    return ANSWERING


async def receive_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the student's answer, resumes the graph, and sends feedback."""
    assert update.message and update.message.text and context.user_data is not None
    user_answer = sanitize_user_input(update.message.text)
    
    # We must explicitly cast to RunnableConfig since user_data holds Any
    import typing
    config = typing.cast(RunnableConfig, context.user_data["config"])

    await update.message.reply_text("🧠 Evaluating your answer...")

    try:
        # Resume graph from interrupt
        tutor_graph.update_state(config, {"user_answer": user_answer})
        state = tutor_graph.invoke(None, config)
        context.user_data["graph_state"] = state
    except Exception as e:
        logger.error(f"Graph resume failed for Telegram user: {e}")
        await update.message.reply_text(
            "❌ Something went wrong grading. Please try again with /start"
        )
        return ConversationHandler.END

    # Get the latest Q&A from history
    history = state.get("questions_history", [])
    if history:
        last_qa = history[-1]
        grade = last_qa.grade
        feedback = last_qa.feedback

        # Grade indicator
        if grade >= 7:
            indicator = f"✅ *Correct!* — Grade: {grade}/10"
        elif grade >= 4:
            indicator = f"⚠️ *Partially Correct* — Grade: {grade}/10"
        else:
            indicator = f"❌ *Incorrect* — Grade: {grade}/10"

        await update.message.reply_text(
            f"{indicator}\n\n📋 *Feedback:* {feedback}",
            parse_mode="Markdown",
        )

    # Check if session is complete
    status = state.get("status", "")
    if status == "done":
        # Send final results
        if history:
            avg_grade = sum(q.grade for q in history) / len(history)
            await update.message.reply_text(
                f"🎉 *Session Complete!*\n\n"
                f"📊 Average Grade: *{avg_grade:.1f}/10*\n"
                f"📝 Questions Answered: *{len(history)}*\n\n"
                f"Send /start to learn something else!",
                parse_mode="Markdown",
            )
        return ConversationHandler.END

    # Send next question
    question = state.get("question", "")
    current_q = state.get("current_question", 1)
    total_q = state.get("total_questions", 1)

    await update.message.reply_text(
        f"📚 *Question {current_q} of {total_q}*\n\n"
        f"{question}\n\n"
        "_Type your answer below:_",
        parse_mode="Markdown",
    )
    return ANSWERING


# ── YouTube Download Handlers ────────────────────────────────────────


async def youtube_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles /youtube command — starts YouTube download flow."""
    assert update.message
    await update.message.reply_text(
        "🎬 *YouTube Transcript Downloader*\n\n"
        "Send me a YouTube video URL and I'll extract the transcript "
        "into the knowledge base.\n\n"
        "📎 _Paste the URL:_",
        parse_mode="Markdown",
    )
    return YOUTUBE_URL


async def receive_youtube_url(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Receives the YouTube URL and asks for an optional title."""
    assert update.message and update.message.text and context.user_data is not None
    url = update.message.text.strip()
    if "youtube.com" not in url and "youtu.be" not in url:
        await update.message.reply_text(
            "⚠️ That doesn't look like a YouTube URL. Please try again:"
        )
        return YOUTUBE_URL

    context.user_data["youtube_url"] = url
    await update.message.reply_text(
        "📝 *Optional:* Give this file a name (or send `.` to use a random ID):",
        parse_mode="Markdown",
    )
    return YOUTUBE_TITLE


async def receive_youtube_title(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Receives the title and downloads the transcript."""
    assert update.message and update.message.text and context.user_data is not None
    title = update.message.text.strip()
    if title == ".":
        title = ""

    url = context.user_data["youtube_url"]
    await update.message.reply_text("⬇️ Downloading transcript...")

    success = download_youtube_to_knowledge(url, custom_title=title)

    if success:
        # Rebuild the vector store with new content
        vsm = get_vector_store_manager()
        vsm.create_and_load_db()

        await update.message.reply_text(
            "✅ *Transcript downloaded and indexed!*\n\n"
            "The knowledge base has been updated. "
            "Send /start to learn from it!",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "❌ Failed to download transcript. "
            "The video might not have captions.\n"
            "Try another URL with /youtube"
        )

    return ConversationHandler.END


# ── Cancel Handler ───────────────────────────────────────────────────


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles /cancel — cancels any active conversation."""
    assert update.message and context.user_data is not None
    await update.message.reply_text(
        "❌ Operation cancelled. Send /start to begin again."
    )
    context.user_data.clear()
    return ConversationHandler.END


# ── Application Builder ─────────────────────────────────────────────


def run_telegram_bot() -> None:
    """Builds and starts the Telegram bot application.

    Reads the ``TELEGRAM_BOT_TOKEN`` from environment variables and
    configures two ConversationHandlers: one for learning sessions
    and one for YouTube downloads.

    Raises:
        ValueError: If ``TELEGRAM_BOT_TOKEN`` is not set or empty.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in .env")
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is missing! "
            "Get one from @BotFather on Telegram and add it to .env"
        )

    logger.info("Starting Telegram bot...")

    app = ApplicationBuilder().token(token).build()

    # Learning session conversation
    learning_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_topic)],
            QUESTION_COUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_question_count)
            ],
            ANSWERING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_answer)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # YouTube download conversation
    youtube_handler = ConversationHandler(
        entry_points=[CommandHandler("youtube", youtube_start)],
        states={
            YOUTUBE_URL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_youtube_url)
            ],
            YOUTUBE_TITLE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_youtube_title)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(learning_handler)
    app.add_handler(youtube_handler)

    logger.success("Telegram bot is running! Press Ctrl+C to stop.")
    app.run_polling()
