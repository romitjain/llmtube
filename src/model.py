import json
import torch
import openai
import logging
import backoff
from typing import Tuple, Dict, List
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .utils import log_time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8"
)
logger = logging.getLogger(__name__)

class Transcriber():

    def __init__(
            self,
            model_id: str = "openai/whisper-large-v3",
            device: str = 'cuda:0',
            torch_dtype: torch.dtype = torch.float16
        ) -> None:

        logger.info(f"Loading the model in memory: {model_id}, {device}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation ="flash_attention_2"
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=2048,
            chunk_length_s=30,
            batch_size=4,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )

    @log_time
    def __call__(self, audio_path:str = None, audio: torch.Tensor = None):
        if audio_path:
            return self.pipe(audio_path)
        
        return self.pipe(audio)

def add_assistant_msg_json(x): return {'role': 'assistant', 'content': json.dumps(x)}
def add_usr_msg_json(x): return {'role': 'user', 'content': json.dumps(x)}
def add_assistant_msg(x): return {'role': 'assistant', 'content': x}
def add_usr_msg(x): return {'role': 'user', 'content': x}


class Summarizer():
    """
    Wrapper over LLM
    """
    def __init__(self, model_id: str, messages: List = [], uri: str = None) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = openai.OpenAI(base_url=uri)
        self.model_id = model_id
        self.messages = messages
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @log_time
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def __call__(
        self,
        message: str,
        json_mode: bool = True,
        **kwargs
    ) -> Tuple[Dict, any]:

        if json_mode:
            message = add_usr_msg_json(message)
        else:
            message = add_usr_msg(message)

        self.messages.append(message)

        logger.debug(f"Messages: {json.dumps(self.messages, indent=4)}")

        op = None
        try:
            if json_mode:
                completions = self.client.chat.completions.create(
                    model=self.model_id,
                    response_format={'type': 'json_object'},
                    messages=self.messages,
                    **kwargs
                )

                if completions.choices[0].finish_reason == 'length':
                    self.messages.pop()
                    raise IOError(
                        f'Reached maximum output length, output format is not reliable. {completions.choices[0].message.content}'
                    )

                op = json.loads(completions.choices[0].message.content)

            else:
                completions = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=self.messages,
                    **kwargs
                )

                op = completions.choices[0].message.content

            logger.debug(f'Prompts: {message}, output: {op}')
            logger.debug(
                f'Tokens used in generation using {self.model_id}: {completions.usage}'
            )

            self.total_tokens += completions.usage.total_tokens
            self.input_tokens += completions.usage.prompt_tokens
            self.output_tokens += completions.usage.completion_tokens

        except Exception as err:
            logger.error(f'Raised error while calling LLM: {err}')

        finally:
            _ = self.messages.pop()
            return op


class VideoSummarizer():
    def __init__(self, model_id: str, messages: List = [], uri: str = None) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = openai.OpenAI(base_url=uri)
        self.model_id = model_id
        self.messages = messages
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def __call__(
        self,
        message: str,
        image: str = None,
        **kwargs
    ) -> Tuple[Dict, any]:

        self._add_usr_msg(message, image)

        completions = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            **kwargs
        )

        op = completions.choices[0].message.content

        logger.debug(f'Prompts: {message}, output: {op}')
        logger.debug(
            f'Tokens used in generation using {self.model_id}: {completions.usage}')

        self._add_assistant_msg(op)

        self.total_tokens += completions.usage.total_tokens
        self.input_tokens += completions.usage.prompt_tokens
        self.output_tokens += completions.usage.completion_tokens

        return op

    def _add_usr_msg(self, msg: str, img: str = None):

        content = [{'type': 'text', 'text': msg}]
        if img:
            content.append({
                'type': 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{img}"}
            })

        self.messages.append({
            'role': 'user',
            'content': content
        })

    def _add_assistant_msg(self, msg: str, role: str = 'assistant'):
        self.messages.append({
            'role': role,
            'content': [{'type': 'text', 'text': msg}]
        })
