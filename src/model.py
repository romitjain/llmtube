import sys
import json
import torch
import openai
import backoff
from loguru import logger
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .utils import log_time

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="DEBUG")

def add_assistant_msg_json(x): return {'role': 'assistant', 'content': json.dumps(x)}
def add_usr_msg_json(x): return {'role': 'user', 'content': json.dumps(x)}
def add_assistant_msg(x): return {'role': 'assistant', 'content': x}
def add_usr_msg(x): return {'role': 'user', 'content': x}

class Transcriber():

    def __init__(
            self,
            model_id: str = "openai/whisper-large-v2",
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

        logger.info("Loaded the model in memory")

    @log_time(name='transcriber')
    def __call__(self, audio_path:str = None, audio: torch.Tensor = None):
        if audio_path:
            return self.pipe(audio_path)
        
        return self.pipe(audio)

class LLM():
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

    @log_time(name='llm')
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


class Embedder():
    def __init__(self, model_id: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_id)

    @log_time(name='embedder')
    def __call__(self, paragraphs: List[str]) -> Dict:

        logger.debug(f"Paragraphs: {paragraphs}")

        paragraph_embeddings = self.model.encode(paragraphs)

        embedding_dict = {}
        for idx, embedding in enumerate(paragraph_embeddings):
            embedding_dict[idx] = embedding

        return embedding_dict

    @log_time(name='clustering')
    def cluster(self, embeddings: Dict, k: int = 5) -> Dict:
        """
        Cluster the embeddings into `k` clusters
        using K means

        Args:
            embeddings (Dict): { idx: embedding }
            k (int, optional): Number of clusters. Defaults to 5.

        Returns:
            Dict: { label: [idxs] }
        """
        num_clusters = min(k, len(embeddings.keys()))
        logger.info(f"Using n={num_clusters} for clustering")

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(list(embeddings.values()))
        cluster_labels = kmeans.labels_
        logger.info(f"Clustering labels: {cluster_labels}")
        logger.info(f"Clustering labels dimension: {cluster_labels.shape}")

        clustering_dict = {}
        for idx, label in enumerate(cluster_labels):
            temp: List = clustering_dict.get(label, [])
            temp.append(idx)

            clustering_dict.update({int(label): temp})

        return clustering_dict
