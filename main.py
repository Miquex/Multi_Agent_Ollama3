"""Multi-Agent Tutor — Main entry point.

Launches both the Streamlit web UI and the Telegram bot
simultaneously when executed.

Usage:
    uv run python main.py
"""

import sys
import subprocess
import threading

from app.utils.logger import logger


def _run_streamlit() -> None:
    """Launches the Streamlit UI in a subprocess."""
    logger.info("Starting Streamlit UI...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"],
        check=False,
    )


def _run_telegram() -> None:
    """Launches the Telegram bot (blocks until stopped)."""
    try:
        from app.ui.telegram_bot import run_telegram_bot

        run_telegram_bot()
    except ValueError as e:
        logger.warning(f"Telegram bot skipped: {e}")
    except Exception as e:
        logger.error(f"Telegram bot crashed: {e}")


def main() -> None:
    """Starts both the Streamlit UI and Telegram bot concurrently.

    Streamlit runs in a background thread so the Telegram bot
    (which uses asyncio polling) can run in the main thread.
    If no TELEGRAM_BOT_TOKEN is set, only Streamlit launches.
    """
    # Run Streamlit in a background thread
    streamlit_thread = threading.Thread(target=_run_streamlit, daemon=True)
    streamlit_thread.start()

    # Run Telegram bot in main thread (it needs the main event loop)
    _run_telegram()

    # If Telegram bot exits (no token), keep alive for Streamlit
    streamlit_thread.join()


if __name__ == "__main__":
    main()
