import time
import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8"
)
logger = logging.getLogger(__name__)

def log_time(func):
    """
    Decorator to log the time taken by a function.

    Args:
        func: The function to be decorated.

    Returns:
        A wrapper function that logs the time taken by the decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")
        return result
    return wrapper

def beautify_metadata(x: Dict) -> str:
    """
    Beautifies a dict into a flat str

    Args:
        x (Dict): Original Dict

    Returns:
        str: Beautified string
    """

    txt = ""

    for k, v in x.items():
        txt += f"{k}: {v}\n"

    return txt
