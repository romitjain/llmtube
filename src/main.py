import os
import sys
import json
import ujson
import argparse
from loguru import logger
from torch.cuda import empty_cache
from textwrap import dedent
from pytube import YouTube
from typing import Dict

from .utils import (
    beautify_metadata,
    log_time,
    get_metadata_video
)
from .model import Transcriber, LLM, Embedder

logger.remove()
logger.add(
    sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO"
)

uri = "http://127.0.0.1:8002/v1/"

def download_audio(yt: YouTube, output_path: str) -> str:
    """
    Downloads the audio from YouTube and saves it locally

    Args:
        yt (YouTube): pytube object
        output_path (str): Output directory

    Returns:
        str: Absolute path of the downloaded audio
    """
    audio_stream = yt.streams.filter(only_audio=True).first()
    out_file = audio_stream.download(output_path=output_path)

    logger.debug(f"Saving audio to: {out_file}")
    base, _ = os.path.splitext(out_file)
    new_file = base + ".mp3"
    logger.debug(f"Renaming audio to: {new_file}")
    os.rename(out_file, new_file)

    return new_file


def summarize(text_map: Dict, metadata: Dict) -> Dict:
    """
    Summarizes complete document by iterating it over
    chunk by chunk

    Args:
        text (Dict): Original text {id: raw text}
        chunk_length_words (int): Number of words to split

    Returns:
        Dict: {id: summary}
    """

    sys_prompt = {
        "role": "system",
        "content": dedent("""
            You are a helpful assistant who summarizes given text.
            The text that will be provided is a transcript of a YouTube video and some metadata about the video.
            You have to provide a concise summary of the transcript.
        """).strip()
    }

    logger.info(f"System Prompt: {sys_prompt['content']}")

    summarizer = LLM(model_id="open-hermes", uri=uri, messages=[sys_prompt])

    summaries_map = {}

    for idx, chunk in text_map.items():
        logger.info(f"Chunk iterator: {idx}")

        chunk_prompt = dedent(f"""
            Original transcript
            {chunk.strip()}
            Metadata of the video
            {beautify_metadata(metadata)}
            Summary
        """).strip()

        logger.debug(f"Chunk prompt: {chunk_prompt}")
        summaries_map[idx] = summarizer(message=chunk_prompt, json_mode=False)

    return summaries_map


def report(text: str, metadata: Dict) -> str:
    """
    Creates a summary of summaries

    Args:
        text (str): Original summaries
        metadata (Dict): Metadata about the video

    Returns:
        str: Summaries of summary
    """
    sys_prompt = {
        "role": "system",
        "content": dedent("""
            You are a helpful assistant who summarizes given text.
            The text that will be provided contains multiple summaries of different parts of an original transcript of the video.
            You have to provide a descriptive summary using the summaries.
        """).strip()
    }

    logger.info(f"System Prompt: {sys_prompt['content']}")

    summarizer = LLM(model_id="open-hermes", uri=uri, messages=[sys_prompt])

    chunk_prompt = dedent(f"""
        Summaries of small chunks
        {text.strip()}
        Metadata of the video
        {beautify_metadata(metadata)}
        Summary
    """).strip()

    logger.debug(chunk_prompt)

    return summarizer(message=chunk_prompt, json_mode=False)


@log_time(name='Complete run')
def main(url: str, output_path: str = "./data/"):
    """
    Steps
    1. Read the audio
    2. Transcribe the audio
    3. Create embedding out of small chunks of data
    4. Cluster the chunks
    5. Concatenate the chunks that are in the same cluster
    6. Summarize each cluster
    7. Summarize the complete video
    """
    os.makedirs(os.path.join(output_path, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "transcript"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "summary"), exist_ok=True)

    ## Step 1: Read the audio
    logger.debug(f"Downloading the audio from: {url}")
    vid = YouTube(url)
    ## Download audio and save locally
    fn = download_audio(vid, os.path.join(output_path, "audio"))
    meta = get_metadata_video(vid)
    logger.debug(f"Metadata of the video: {ujson.dumps(meta)}")

    _, audio_fn = os.path.split(fn)
    video_fn, _ = os.path.splitext(audio_fn)

    # Step 2: Transcribe the audio
    transcriber = Transcriber()
    transcript = transcriber(audio_path=fn)
    empty_cache()
    del transcriber

    transcript_fn = os.path.join(output_path, "transcript", video_fn + ".txt")
    logger.info(f"Saving transcript to: {transcript_fn}")

    ## Save the transcript locally
    with open(transcript_fn, "w") as fp:
        fp.write(ujson.dumps(transcript, indent=4))

    ## Step 3: Create embedding out of small chunks of data
    ## Step 4: Cluster the chunks
    chunks = transcript["text"].split(".")
    chunk_length_sentence = 10
    paragraphs = []

    for chunk_iterator in range(0, len(chunks), chunk_length_sentence):
        logger.info(f"Chunk iterator: {chunk_iterator}")
        chunk = ". ".join(chunks[chunk_iterator:min(chunk_iterator+chunk_length_sentence, len(chunks))])
        paragraphs.append(chunk)

    embedder = Embedder()
    transcript_embeddings = embedder(paragraphs)
    transcript_clusters = embedder.cluster(transcript_embeddings)

    ## Step 5: Concatenate the chunks that are in the same cluster
    concat_transcript = {}
    # For each label by the clustering, concatenate the raw transcript
    for k, v in transcript_clusters.items():
        for s in v:
            concat_transcript[k] = concat_transcript.get(k, "") + '\n' + paragraphs[s]

    ## Step 6: Summarize each cluster
    summaries = summarize(concat_transcript, metadata=meta)

    ## Step 7: Complete summary
    complete_summary = report("\n\n".join(list(summaries.values())), meta)

    summaries[-1] = complete_summary

    summary_fn = os.path.join(output_path, "summary", video_fn + ".txt")
    logger.info(f"Saving summary to: {summary_fn}")

    # Save the summary locally
    with open(summary_fn, "w") as fp:
        json.dump(summaries, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a URL')
    parser.add_argument('url', type=str, help='URL to process')

    args = parser.parse_args()

    main(url=args.url)
