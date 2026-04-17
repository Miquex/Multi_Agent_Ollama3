"""YouTube transcript fetcher for the RAG knowledge base.

Downloads transcripts from individual videos or entire playlists
and saves them as text files in the knowledge folder for indexing.
"""

import os
import pathlib
import uuid

from pytube import Playlist  # type: ignore[import-untyped]
from langchain_community.document_loaders import YoutubeLoader

from app.utils.logger import logger


def download_youtube_to_knowledge(
    youtube_url: str,
    custom_title: str = "",
) -> bool:
    """Downloads a single YouTube video transcript and saves it as a text file.

    Uses LangChain's YoutubeLoader to extract the transcript, then writes
    it to the knowledge folder for subsequent RAG indexing.

    Args:
        youtube_url: Full YouTube video URL to download the transcript from.
        custom_title: Optional human-readable file name. If empty, a random
            ID is generated instead.

    Returns:
        True if the transcript was successfully saved, False otherwise.
    """
    logger.info(f"Connecting to YouTube: {youtube_url}")

    try:
        loader = YoutubeLoader.from_youtube_url(
            youtube_url,
            add_video_info=False,
        )
        docs = loader.load()
    except Exception as e:
        logger.error(
            f"Failed to fetch transcript. The video might not have "
            f"CC closed captions. Error: {e!s}"
        )
        return False

    if not docs:
        logger.warning("No transcript found in video")
        return False

    if custom_title:
        video_title = custom_title.replace(" ", "_")
    else:
        random_id = uuid.uuid4().hex[:6]
        video_title = f"YT_Video_{random_id}"

    transcript = docs[0].page_content

    knowledge_folder = (
        pathlib.Path(__file__).resolve().parent.parent / "data" / "knowledge"
    )
    os.makedirs(knowledge_folder, exist_ok=True)

    file_path = knowledge_folder / f"{video_title}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    logger.success(f"YouTube video successfully saved as text: {file_path}")
    return True


def download_youtube_playlist(playlist_url: str) -> None:
    """Downloads transcripts for every video in a YouTube playlist.

    Iterates through all video URLs in the playlist and delegates each
    to ``download_youtube_to_knowledge`` for individual processing.

    Args:
        playlist_url: Full YouTube playlist URL.
    """
    logger.info(f"Connecting to YouTube Playlist: {playlist_url}")
    try:
        playlist = Playlist(playlist_url)
        logger.info(
            f"Found {len(playlist.video_urls)} videos in playlist: "
            f"{playlist.title}. Downloading transcripts..."
        )

        for video_url in playlist.video_urls:
            download_youtube_to_knowledge(video_url)
        logger.success(
            "Playlist completely downloaded and loaded into knowledge folder!"
        )

    except Exception as e:
        logger.error(
            f"Failed to read playlist. Check if the URL is correct "
            f"or private. Error: {e!s}"
        )
