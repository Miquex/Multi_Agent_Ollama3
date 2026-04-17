"""Logger configuration for the Multi-Agent Tutor application.

Sets up Loguru with both console (stderr) and file sinks.
The file sink rotates at 10 MB and retains compressed logs for 5 days.
"""

import sys
import pathlib

from loguru import logger


def setup_logger() -> None:
    """Configures the global Loguru logger with console and file outputs.

    Console output is colorized and set to INFO level.
    File output includes all levels from DEBUG and above, with
    automatic rotation and zip compression.
    """
    logger.remove()
    log_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    logger.add(
        log_dir / "logs.log",
        rotation="10 MB",
        compression="zip",
        retention="5 days",
        level="DEBUG",
    )


setup_logger()
