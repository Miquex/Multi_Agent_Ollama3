"""Streamlit UI for the Multi-Agent Tutor application.

Provides the interactive front-end for the tutor system including:
- Topic input and question count configuration
- Summary display after RAG retrieval
- Question-by-question interaction with grade feedback
- Knowledge base file viewer
- YouTube transcript downloader
"""

import uuid
import pathlib

import streamlit as st

from langchain_core.runnables import RunnableConfig
from app.core.graph import tutor_graph
from app.utils.youtube_fetcher import (
    download_youtube_to_knowledge,
    download_youtube_playlist,
)
from app.core.rag.vector_store_data import get_vector_store_manager
from app.utils.sanitizer import sanitize_topic, sanitize_user_input

st.set_page_config(page_title="Agentic AI Tutor", layout="wide")
st.title("🎓 Multi-Agent Tutor")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

left_col, right_col = st.columns([2, 1])

with left_col:
    topic = st.text_input(
        "What would you like to learn today?",
        placeholder="e.g. Docker, Python, etc.",
    )
    num_questions = st.slider(
        "How many questions do you want?",
        min_value=1,
        max_value=10,
        value=3,
    )

    if st.button("🚀 Start Learning") and topic:
        # Always start a fresh session with a new thread_id
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.show_feedback = False
        if "graph_state" in st.session_state:
            del st.session_state.graph_state

        clean_topic = sanitize_topic(topic)
        config: RunnableConfig = {"configurable": {"thread_id": str(st.session_state.thread_id)}}
        with st.spinner(
            "Agents are researching, summarizing, and generating a question..."
        ):
            state = tutor_graph.invoke(
                {
                    "topic": clean_topic,
                    "session_id": st.session_state.thread_id,
                    "total_questions": num_questions,
                    "current_question": 0,
                    "questions_history": [],
                },
                config,
            )
            st.session_state.graph_state = state
            st.rerun()

    # Build config from current thread_id
    config = {"configurable": {"thread_id": str(st.session_state.thread_id)}}

    # Display results based on session state
    if "graph_state" in st.session_state:
        state = st.session_state.graph_state

        # Always show the summary if available
        if state.get("summary"):
            with st.expander("📝 Topic Summary — Key Ideas", expanded=True):
                st.markdown(state["summary"])

        # Show history of past Q&A pairs (collapsed)
        history = state.get("questions_history", [])
        if history:
            st.divider()
            st.subheader("📊 Your Progress")
            for i, qa in enumerate(history, 1):
                is_latest = (i == len(history)) and st.session_state.show_feedback
                with st.expander(
                    f"Question {i} — Grade: {qa.grade}/10",
                    expanded=is_latest,
                ):
                    st.markdown(f"**Question:** {qa.question}")
                    st.markdown(f"**Your Answer:** {qa.user_answer}")
                    st.markdown(f"**Grade:** {qa.grade} / 10")
                    st.markdown(f"**Feedback:** {qa.feedback}")

        # --- FEEDBACK SCREEN: show after each answer before next question ---
        if st.session_state.show_feedback and history:
            last_qa = history[-1]
            st.divider()

            # Show correct/incorrect indicator
            if last_qa.grade >= 7:
                st.success(f"✅ Correct! — Grade: {last_qa.grade}/10")
            elif last_qa.grade >= 4:
                st.warning(f"⚠️ Partially Correct — Grade: {last_qa.grade}/10")
            else:
                st.error(f"❌ Incorrect — Grade: {last_qa.grade}/10")

            st.markdown(f"**Tutor Feedback:** {last_qa.feedback}")

            # Show "Next Question" or "See Results" depending on status
            if state.get("status") == "waiting_for_user":
                if st.button("➡️ Next Question"):
                    st.session_state.show_feedback = False
                    st.rerun()
            elif state.get("status") == "done":
                if st.button("📊 See Final Results"):
                    st.session_state.show_feedback = False
                    st.rerun()

        # --- QUESTION SCREEN: show current question ---
        elif (
            state.get("status") == "waiting_for_user"
            and not st.session_state.show_feedback
        ):
            st.divider()
            current_q = state.get("current_question", 1)
            total_q = state.get("total_questions", 1)
            st.info(f"📚 Question {current_q} of {total_q}")
            st.markdown(f"### {state['question']}")

            user_answer = st.text_area("Your Answer:", key=f"answer_{current_q}")

            if st.button("✅ Submit Answer"):
                with st.spinner("Agent 3 (Evaluator) is grading..."):
                    # Sanitize and inject the answer into checkpointed state,
                    # then resume the graph from the interrupt point.
                    clean_answer = sanitize_user_input(user_answer)
                    tutor_graph.update_state(config, {"user_answer": clean_answer})
                    final_state = tutor_graph.invoke(None, config)
                    st.session_state.graph_state = final_state
                    st.session_state.show_feedback = True
                    st.rerun()

        # --- FINAL RESULTS SCREEN ---
        elif state.get("status") == "done" and not st.session_state.show_feedback:
            st.divider()
            st.success("🎉 Session Complete!")

            # Calculate average grade
            if history:
                avg_grade = sum(q.grade for q in history) / len(history)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Average Grade", value=f"{avg_grade:.1f} / 10")
                with col2:
                    st.metric(label="Questions Answered", value=f"{len(history)}")

            if st.button("🔄 Learn Something Else"):
                del st.session_state.graph_state
                st.session_state.show_feedback = False
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()


with right_col:
    st.subheader("📄 RAG Sources")
    if "graph_state" in st.session_state:
        sources = st.session_state.graph_state.get("source_documents", [])
        if sources:
            st.write("Agent 1 used these documents from the Vector DB:")
            for source in set(sources):
                st.markdown(f"- `{source}`")
        else:
            st.write("No RAG documents found")

    # Knowledge Base Viewer
    st.divider()
    st.subheader("📂 Knowledge Base")
    knowledge_path = (
        pathlib.Path(__file__).resolve().parent.parent / "data" / "knowledge"
    )
    if knowledge_path.exists():
        files = sorted(knowledge_path.iterdir())
        knowledge_files = [f for f in files if f.is_file()]
        if knowledge_files:
            st.write(f"**{len(knowledge_files)} document(s) loaded:**")
            for f in knowledge_files:
                size_kb = f.stat().st_size / 1024
                with st.expander(f"📄 {f.name} ({size_kb:.1f} KB)"):
                    try:
                        content = f.read_text(encoding="utf-8")
                        preview = content[:1000]
                        if len(content) > 1000:
                            preview += "\n\n... (truncated)"
                        st.text(preview)
                    except Exception:
                        st.write("*(binary file — cannot preview)*")
        else:
            st.info("No documents yet. Use the sidebar to add YouTube content!")
    else:
        st.info("Knowledge folder not created yet. Feed some data first!")


with st.sidebar:
    st.header("🎬 Feed the AI YouTube Data")
    st.markdown("Paste a single video or an entire playlist URL here:")
    yt_url = st.text_input("YouTube URL:")
    yt_title = st.text_input("Optional: Name this file (leave blank for random ID):")

    if st.button("⬇️ Download & Process"):
        with st.spinner("Executing download and updating Database..."):
            if "playlist" in yt_url.lower():
                download_youtube_playlist(yt_url)
            elif yt_url:
                download_youtube_to_knowledge(yt_url, custom_title=yt_title)

            vsm = get_vector_store_manager()
            vsm.create_and_load_db()

            st.success("Database fully trained on YouTube content!")
