import os
import typer
import ujson
import logging
from torch.cuda import empty_cache
from textwrap import dedent
from pytube import YouTube
from typing import List, Tuple, Dict

from .utils import beautify_metadata
from .model import Transcriber, Summarizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S", encoding="utf-8"
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def get_metadata_video(yt: YouTube) -> Dict:
    """
    Gets metadata about the given YouTube video

    Args:
        yt (YouTube): pytube object

    Returns:
        Dict: {
            "title": "xxx",
            "len": "xxx",
            "tags": "xxx",
            "channel": "xxx"
        }
    """
    return {
        "title": yt.title,
        #"length": yt.length,
        "tags": yt.keywords,
        "channel": yt.author
    }


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


def summarize(text: str, metadata: Dict, chunk_length_sentence: int = 10) -> List[Tuple]:
    """
    Summarizes complete document by iterating it over
    chunk by chunk

    Args:
        text (str): Original text
        chunk_length_words (int): Number of words to split

    Returns:
        List[Tuple]: Each element of the list is a tuple where first element
        is the original text and the second element is the summarized text
    """
    logger.info(
        f"Starting summarizer with chunk size: {chunk_length_sentence} sentences"
    )

    sys_prompt = {
        "role": "system",
        "content": dedent("""
            You are a helpful assistant who summarizes given text.
            The text that will be provided is a transcript of a YouTube video and some metadata about the video.
            You have to provide a concise summary of the transcript.
        """).strip()
    }

    logger.info(f"System Prompt: {sys_prompt['content']}")

    uri = "http://127.0.0.1:8002/v1/"
    summarizer = Summarizer(model_id="open-hermes", uri=uri, messages=[sys_prompt])

    summaries = []
    chunks = text.split(".")

    for chunk_iterator in range(0, len(chunks), chunk_length_sentence):
        logger.info(f"Chunk iterator: {chunk_iterator}")
        chunk = ". ".join(chunks[chunk_iterator:min(chunk_iterator+chunk_length_sentence, len(chunks))])

        chunk_prompt = dedent(f"""Original transcript
            {chunk.strip()}
            Metadata of the video
            {beautify_metadata(metadata)}
            Summary
        """).strip()

        logger.info(f"Chunk prompt: {chunk_prompt}")
        summaries.append(
            summarizer(message=chunk_prompt, json_mode=False)
        )

    return summaries

def main(url: str, output_path: str = "./data/"):
    os.makedirs(os.path.join(output_path, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "transcript"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "summary"), exist_ok=True)

    logger.debug(f"Downloading the audio from: {url}")
    vid = YouTube(url)
    ## Download audio and save locally
    fn = download_audio(vid, os.path.join(output_path, "audio"))
    meta = get_metadata_video(vid)
    logger.debug(f"Metadata of the video: {ujson.dumps(meta)}")

    _, audio_fn = os.path.split(fn)
    video_fn, _ = os.path.splitext(audio_fn)

    ## Transcribe the audio
    transcriber = Transcriber()
    transcript = transcriber(audio_path=fn)
    empty_cache()
    del transcriber

    transcript_fn = os.path.join(output_path, "transcript", video_fn + ".txt")
    logger.info(f"Saving transcript to: {transcript_fn}")

    ## Save the transcript locally
    with open(transcript_fn, "w") as fp:
        fp.write(ujson.dumps(transcript, indent=4))

    list_of_summaries = summarize(transcript["text"], metadata=meta)
    summary_fn = os.path.join(output_path, "summary", video_fn + ".txt")
    logger.info(f"Saving summary to: {summary_fn}")

    # Save the summary locally
    with open(summary_fn, "w") as fp:
        for n in list_of_summaries:
            if not n:
                continue
            fp.write(n)
            fp.write('\n')

if __name__ == "__main__":
    typer.run(main)



