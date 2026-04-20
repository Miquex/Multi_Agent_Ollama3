# ── Stage 1: Builder ─────────────────────────────────────────────────
# Installs all dependencies in an isolated layer so the final image
# only carries the compiled virtual-environment, not the build tools.
FROM python:3.13-slim AS builder

# Install uv — the ultra-fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# Copy only dependency manifests first (leverages Docker layer cache —
# dependencies are only re-installed when pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./

# Install production dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy the full source code
COPY . .

# Install the project itself (editable-style, into the same venv)
RUN uv sync --frozen --no-dev


# ── Stage 2: Runtime ─────────────────────────────────────────────────
# Minimal image with only the Python runtime and installed packages.
FROM python:3.13-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 tutor \
    && useradd --uid 1000 --gid tutor --create-home tutor

WORKDIR /app

# Copy the complete virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application source code
COPY --from=builder /build/app ./app
COPY --from=builder /build/main.py ./main.py

# Ensure the data directories exist (they are volumes at runtime)
RUN mkdir -p app/data/knowledge app/data/chroma_db app/data/logs \
    && chown -R tutor:tutor /app

# Put the virtual-environment's Python first on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Streamlit default port
EXPOSE 8501

# Switch to non-root user
USER tutor

# Health check: ping the Streamlit server every 30 s
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Launch both Streamlit + Telegram bot via the main entry point
CMD ["python", "main.py"]
