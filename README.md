# 🎓 Multi-Agent RAG Tutor

An AI-powered tutoring system that uses multiple specialized agents orchestrated via **LangGraph** to deliver personalized learning sessions. The system retrieves knowledge from a local document store (RAG), generates summaries, asks questions, and evaluates student answers — all powered by a configurable LLM backend.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (Frontend)                     │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────────┐  │
│  │  Topic   │ │  Question    │ │ Feedback │ │  Knowledge    │  │
│  │  Input   │ │  Display     │ │ Screen   │ │  Base Viewer  │  │
│  └──────────┘ └──────────────┘ └──────────┘ └───────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LangGraph State Machine (Core)                  │
│                                                                 │
│   START ──▶ Researcher ──▶ Summarizer ──▶ Examiner             │
│                                              │                  │
│                                        ┌─────┴─────┐           │
│                                        │ interrupt  │           │
│                                        │ (wait for  │           │
│                                        │  answer)   │           │
│                                        └─────┬─────┘           │
│                                              ▼                  │
│                               Evaluator ──▶ Loop? ──▶ END      │
│                                   ▲            │                │
│                                   └────────────┘                │
│                                  (next question)                │
└─────────────────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │  ChromaDB  │ │  Ollama  │ │  Prompt      │
   │  (RAG)     │ │  (LLM)   │ │  Templates   │
   └────────────┘ └──────────┘ └──────────────┘
```

## Agent Roles

| Agent | Role | Input | Output |
|---|---|---|---|
| **Researcher** | Retrieves relevant documents from ChromaDB | Topic | Context + Sources |
| **Summarizer** | Creates a structured summary of key ideas | Context | Summary |
| **Examiner** | Generates comprehension questions from the summary | Summary | Question |
| **Evaluator** | Grades student answers and provides feedback | Answer + Context | Grade + Feedback |

## Tech Stack

| Component | Technology |
|---|---|
| **Orchestration** | LangGraph (StateGraph with checkpointing) |
| **LLM** | Any OpenAI-compatible API (Ollama, OpenAI, etc.) |
| **RAG** | ChromaDB + HuggingFace Embeddings (`all-MiniLM-L6-v2`) |
| **UI** | Streamlit |
| **Models** | Pydantic v2 (typed state + structured output) |
| **Logging** | Loguru (console + rotating file) |
| **Testing** | pytest |

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally (or any OpenAI-compatible API)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Multi_Agent_Ollama3.git
cd Multi_Agent_Ollama3

# Install dependencies
uv sync

# Pull an LLM model (if using Ollama)
ollama pull llama3.1
```

### Configuration

Create a `.env` file in the project root:

```env
MODEL_NAME=llama3.1
MODEL_BASE_URL=http://localhost:11434/v1
API_KEY=ollama
```

### Add Knowledge

Place `.pdf`, `.md`, or `.txt` files in `app/data/knowledge/`, or use the YouTube downloader in the UI sidebar.

### Run

```bash
# Start Ollama (in a separate terminal)
ollama serve

# Launch the application
uv run python main.py
```

## Project Structure

```
Multi_agent/
├── main.py                          # Entry point — launches Streamlit
├── pyproject.toml                   # Dependencies & tool config
├── .env                             # LLM configuration (gitignored)
├── app/
│   ├── core/
│   │   ├── agents/
│   │   │   └── tutor_agents.py      # 4 agent node functions
│   │   ├── rag/
│   │   │   └── vector_store_data.py # ChromaDB manager (singleton)
│   │   ├── graph.py                 # LangGraph state machine
│   │   ├── llm_factory.py           # Cached LLM client factory
│   │   └── prompts.py               # All prompt templates
│   ├── ui/
│   │   └── streamlit_app.py         # Streamlit frontend
│   ├── utils/
│   │   ├── logger.py                # Loguru configuration
│   │   ├── pydantic_config.py       # Pydantic state & output models
│   │   ├── sanitizer.py             # Input sanitization
│   │   └── youtube_fetcher.py       # YouTube transcript downloader
│   └── data/
│       ├── knowledge/               # Source documents (gitignored)
│       ├── chroma_db/               # Vector store (gitignored)
│       └── logs/                    # Application logs
└── tests/
    ├── conftest.py                  # Shared fixtures
    ├── test_models.py               # Pydantic model tests
    ├── test_sanitizer.py            # Sanitizer tests
    ├── test_llm_factory.py          # LLM factory tests
    ├── test_prompts.py              # Prompt template tests
    └── test_graph.py                # Graph routing tests
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Design Decisions

- **Model-Agnostic:** Uses `PydanticOutputParser` instead of `with_structured_output()` for compatibility with any LLM backend. Fallback to plain-text if JSON parsing fails.
- **Input Sanitization:** User inputs are sanitized against common prompt injection patterns before reaching LLM prompts.
- **Singleton Pattern:** `VectorStoreManager` and `get_llm()` are cached to avoid re-initializing heavy resources (embedding model, LLM client) on every request.
- **Document Chunking:** `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) ensures better retrieval quality from the vector store.

## License

MIT
