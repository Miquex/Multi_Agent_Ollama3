"""Multi-Agent Tutor — Main entry point.

Launches the Streamlit-based tutor UI.
Run with: uv run python main.py
"""

import subprocess
import sys


def main():
    """Start the Multi-Agent Tutor Streamlit application."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
