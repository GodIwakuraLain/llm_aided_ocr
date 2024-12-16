import os
import glob
import traceback
import tempfile
import asyncio
import tempfile
import json
import re
import time
import urllib.request
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import List, Dict, Tuple, Optional
from pdf2image import convert_from_path
import pytesseract
from llama_cpp import Llama, LlamaGrammar
import tiktoken
import numpy as np
from PIL import Image
from decouple import Config as DecoupleConfig, RepositoryEnv
import cv2
from filelock import FileLock, Timeout
from transformers import AutoTokenizer
import openai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configuration
config = DecoupleConfig(RepositoryEnv('.env'))

USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", default=False, cast=bool)
API_PROVIDER = config.get("API_PROVIDER", default="OPENAI", cast=str)  # OPENAI or CLAUDE
ANTHROPIC_API_KEY = config.get("ANTHROPIC_API_KEY", default="your-anthropic-api-key", cast=str)
OPENAI_API_KEY = config.get("OPENAI_API_KEY", default="your-openai-api-key", cast=str)
OPENAI_BASE_URL = config.get("OPENAI_BASE_URL", default="https://api.openai.com/v1", cast=str)
CLAUDE_MODEL_STRING = config.get("CLAUDE_MODEL_STRING", default="claude-3-haiku-20240307", cast=str)
CLAUDE_MAX_TOKENS = 4096  # Maximum allowed tokens for Claude API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 300  # Don't use the full max tokens to avoid hitting the limit
OPENAI_COMPLETION_MODEL = config.get("OPENAI_COMPLETION_MODEL", default="gpt-4o-mini", cast=str)
OPENAI_EMBEDDING_MODEL = config.get("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small", cast=str)
OPENAI_MAX_TOKENS = config.get("OPENAI_MAX_TOKENS", default="12000", cast=int)  # Maximum allowed tokens for OpenAI API
DEFAULT_LOCAL_MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS = 2048
input_pdf_file_path = config.get("INPUT_PDF_FILE_PATH", default="input.pdf", cast=str)
TEMP_IMAGE_DIR = config.get("TEMP_IMAGE_DIR", default=".temp_pdf_images", cast=str)
USE_VERBOSE = False
ASYNC_API_REQUESTS = config.get("ASYNC_API_REQUESTS", default=True, cast=bool)  # New configuration for asynchronous/synchronous API requests

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU Check
def is_gpu_available():
    """Checks if GPU is available and logs relevant information."""
    if not GPU_AVAILABLE:
        logging.warning("GPU support not available: nvgpu module not found")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0,
                "error": "nvgpu module not found"}
    try:
        gpu_info = nvgpu.gpu_info()
        num_gpus = len(gpu_info)
        if num_gpus == 0:
            logging.warning("No GPUs found on the system")
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
        first_gpu_vram = gpu_info[0]['mem_total']
        total_vram = sum(gpu['mem_total'] for gpu in gpu_info)
        logging.info(f"GPU(s) found: {num_gpus}, Total VRAM: {total_vram} MB")
        return {"gpu_found": True, "num_gpus": num_gpus, "first_gpu_vram": first_gpu_vram, "total_vram": total_vram,
                "gpu_info": gpu_info}
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}

# Model Download
async def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    """Downloads the LLM model if it doesn't exist."""
    download_status = []
    model_url = "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
    model_name = os.path.basename(model_url)
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')

    os.makedirs(models_dir, exist_ok=True)
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    status = {"url": model_url, "status": "success", "message": "File already exists."}
    filename = os.path.join(models_dir, model_name)

    try:
        with lock.acquire(timeout=1200):  # Timeout after 20 minutes
            if not os.path.exists(filename):
                logging.info(f"Downloading model {model_name} from {model_url}...")
                urllib.request.urlretrieve(model_url, filename)
                file_size = os.path.getsize(filename) / (1024 * 1024)
                if file_size < 100:
                    os.remove(filename)
                    status["status"] = "failure"
                    status["message"] = f"Downloaded file is too small ({file_size:.2f} MB), probably not a valid model file."
                    logging.error(f"Error: {status['message']}")
                else:
                    logging.info(f"Successfully downloaded: {filename} (Size: {file_size:.2f} MB)")
            else:
                logging.info(f"Model file already exists: {filename}")
    except Timeout:
        logging.error(f"Error: Could not acquire lock for downloading {model_name}")
        status["status"] = "failure"
        status["message"] = "Could not acquire lock for downloading."

    download_status.append(status)
    logging.info("Model download process completed.")
    return [model_name], download_status

# Model Loading
def load_model(llm_model_name: str, raise_exception: bool = True):
    """Loads the LLM model with GPU acceleration if available, otherwise falls back to CPU."""
    global USE_VERBOSE
    try:
        current_file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_file_path)
        models_dir = os.path.join(base_dir, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logging.error(f"Error: No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        model_file_path = max(matching_files, key=os.path.getmtime)  # Select the most recently modified file
        logging.info(f"Loading model: {model_file_path}")
        try:
            logging.info("Attempting to load model with GPU acceleration...")
            model_instance = Llama(
                model_path=model_file_path,
                n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                verbose=USE_VERBOSE,
                n_gpu_layers=-1  # Offload all layers to the GPU
            )
            logging.info("Model loaded successfully with GPU acceleration.")
        except Exception as gpu_e:
            logging.warning(f"Failed to load model with GPU acceleration: {gpu_e}")
            logging.info("Falling back to CPU...")
            try:
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=0  # No GPU layers
                )
                logging.info("Model loaded successfully with CPU.")
            except Exception as cpu_e:
                logging.error(f"Failed to load model with CPU: {cpu_e}")
                if raise_exception:
                    raise
                return None
        return model_instance
    except Exception as e:
        logging.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_exception:
            raise
        return None

# API Interaction Functions
async def generate_completion(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    """Generates text completion based on the selected API provider or local LLM."""
    if USE_LOCAL_LLM:
        return await generate_completion_from_local_llm(DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens)
    elif API_PROVIDER == "OPENAI":
        if ASYNC_API_REQUESTS:
            return await generate_completion_from_openai(prompt, max_tokens)
        else:
            return generate_completion_from_openai_sync(prompt, max_tokens)
    else:
        logging.error(f"Invalid API_PROVIDER: {API_PROVIDER}")
        return None

def get_tokenizer(model_name: str):
    """Gets the appropriate tokenizer based on the model name."""
    if model_name.lower().startswith("gpt-"):
        return tiktoken.encoding_for_model(model_name)
    elif model_name.lower().startswith("claude-"):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    elif model_name.lower().startswith("llama-"):
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b", clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def estimate_tokens(text: str, model_name: str) -> int:
    """Estimates the number of tokens in a given text using the specified model's tokenizer."""
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Error using tokenizer for {model_name}: {e}. Falling back to approximation.")
        return approximate_tokens(text)

def approximate_tokens(text: str) -> int:
    """Approximates the number of tokens in a given text based on common patterns."""
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    tokens = re.findall(r'\b\w+\b|\S', text)  # Split on whitespace and punctuation, keeping punctuation
    count = 0
    for token in tokens:
        if token.isdigit():
            count += max(1, len(token) // 2)  # Numbers often tokenize to multiple tokens
        elif re.match(r'^[A-Z]{2,}$', token):  # Acronyms
            count += len(token)
        elif re.search(r'[^\w\s]', token):  # Punctuation and special characters
            count += 1
        elif len(token) > 10:  # Long words often split into multiple tokens
            count += len(token) // 4 + 1
        else:
            count += 1
    return int(count * 1.1)  # Add a 10% buffer for potential underestimation

def chunk_text(text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
    """Splits the text into chunks that fit within the specified token limit, considering sentence boundaries."""
    chunks = []
    tokenizer = get_tokenizer(model_name)
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split into sentences
    current_chunk = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_chunk_tokens + sentence_tokens > max_chunk_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    adjusted_chunks = adjust_overlaps(chunks, tokenizer, max_chunk_tokens)
    return adjusted_chunks

def split_long_sentence(sentence: str, max_tokens: int, model_name: str) -> List[str]:
    """Splits a long sentence into smaller chunks based on word boundaries."""
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    tokenizer = get_tokenizer(model_name)

    for word in words:
        word_tokens = len(tokenizer.encode(word))
        if current_chunk_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def adjust_overlaps(chunks: List[str], tokenizer, max_chunk_tokens: int, overlap_size: int = 50) -> List[str]:
    """Adjusts the chunks to include overlaps between consecutive chunks."""
    adjusted_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            adjusted_chunks.append(chunks[i])
        else:
            overlap_tokens = len(tokenizer.encode(' '.join(chunks[i - 1].split()[-overlap_size:])))
            current_tokens = len(tokenizer.encode(chunks[i]))
            if overlap_tokens + current_tokens > max_chunk_tokens:
                overlap_adjusted = chunks[i].split()[:-overlap_size]
                adjusted_chunks.append(' '.join(overlap_adjusted))
            else:
                adjusted_chunks.append(' '.join(chunks[i - 1].split()[-overlap_size:] + chunks[i].split()))

    return adjusted_chunks

async def generate_completion_from_claude(prompt: str, max_tokens: int = CLAUDE_MAX_TOKENS - TOKEN_BUFFER) -> Optional[str]:
    """Generates text completion using the Claude API."""
    if not ANTHROPIC_API_KEY:
        logging.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return None
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    prompt_tokens = estimate_tokens(prompt, CLAUDE_MODEL_STRING)
    adjusted_max_tokens = min(max_tokens, CLAUDE_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for Claude API. Chunking the input.")
        chunks = chunk_text(prompt, CLAUDE_MAX_TOKENS - TOKEN_CUSHION, CLAUDE_MODEL_STRING)
        results = []
        for chunk in chunks:
            try:
                async with client.messages.stream(
                        model=CLAUDE_MODEL_STRING,
                        max_tokens=CLAUDE_MAX_TOKENS // 2,
                        temperature=0.7,
                        messages=[{"role": "user", "content": chunk}],
                ) as stream:
                    message = await stream.get_final_message()
                    results.append(message.content[0].text)
                    logging.info(
                        f"Chunk processed. Input tokens: {message.usage.input_tokens:,}, Output tokens: {message.usage.output_tokens:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        try:
            async with client.messages.stream(
                    model=CLAUDE_MODEL_STRING,
                    max_tokens=adjusted_max_tokens,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = await stream.get_final_message()
                output_text = message.content[0].text
                logging.info(f"Total input tokens: {message.usage.input_tokens:,}")
                logging.info(f"Total output tokens: {message.usage.output_tokens:,}")
                logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
                return output_text
        except Exception as e:
            logging.error(f"An error occurred while requesting from Claude API: {e}")
            return None

async def generate_completion_from_openai(prompt: str, max_tokens: int = 5000) -> str:
    """Generates text completion using the OpenAI API asynchronously."""
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return ""

    prompt_tokens = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    adjusted_max_tokens = min(max_tokens, OPENAI_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)

    if adjusted_max_tokens <= 0:
        logging.warning(
            "Prompt is too long for the API (consider increasing OPENAI_MAX_TOKENS or improving chunking). Chunking the input."
        )
        chunks = chunk_text(prompt, OPENAI_MAX_TOKENS - TOKEN_CUSHION, OPENAI_COMPLETION_MODEL)
        results = []
        for chunk in chunks:
            # 使用重试机制处理每个 chunk
            result = await retry_openai_request(chunk, adjusted_max_tokens)
            if not result:
                logging.error("Failed to process a chunk after multiple retries.")
                return ""  # Return empty string on error after retries
            results.append(result)
        return " ".join(results)
    else:       
        return await retry_openai_request(prompt, adjusted_max_tokens)

def generate_completion_from_openai_sync(prompt: str, max_tokens: int = 5000) -> str:
    """Generates text completion using the OpenAI API synchronously."""
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return ""
    
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    prompt_tokens = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    adjusted_max_tokens = min(max_tokens, OPENAI_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)

    if adjusted_max_tokens <= 0:
        logging.warning(
            "Prompt is too long for the API (consider increasing OPENAI_MAX_TOKENS or improving chunking). Chunking the input."
        )
        chunks = chunk_text(prompt, OPENAI_MAX_TOKENS - TOKEN_CUSHION, OPENAI_COMPLETION_MODEL)
        results = []
        for chunk in chunks:            
            result = retry_openai_request_sync(client, chunk, adjusted_max_tokens)
            if not result:
                logging.error("Failed to process a chunk after multiple retries.")
                return ""  # Return empty string on error after retries
            results.append(result)
        return " ".join(results)
    else:        
        return retry_openai_request_sync(client, prompt, adjusted_max_tokens)

def retry_openai_request_sync(client, prompt: str, adjusted_max_tokens: int, retries: int = 3, delay: int = 10) -> str:
    """Helper function to handle retries for synchronous OpenAI API requests."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_COMPLETION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
            )

            # Compatibility handling: extract content from response
            if isinstance(response, str):
                output_text = response
            else:
                output_text = response.choices[0].message.content

            if not output_text:
                logging.error("Received empty response from api.")
                return ""

            logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
            return output_text

        except Exception as e:
            if isinstance(e, openai.APIError):  # 针对所有 openai.APIError 进行重试
                logging.warning(f"OpenAI API error: {e}, attempt {attempt + 1} of {retries}. Retrying in {delay} seconds...")
            elif isinstance(e, Exception): # catch other types of exceptions
                logging.error(f"An error occurred (attempt {attempt+1}): {e}")
            
            if attempt < retries - 1:
                time.sleep(delay)  # Use time.sleep for synchronous requests
            else:
                logging.error(f"Request failed after {retries} attempts.")
                return ""  # Return empty string on error after retries

async def retry_openai_request(prompt: str, adjusted_max_tokens: int, retries: int = 3, delay: int = 10) -> str:
    """Helper function to handle retries for OpenAI API requests."""
    for attempt in range(retries):
        try:
            response = await openai_client.chat.completions.create(
                model=OPENAI_COMPLETION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
            )

            # Compatibility handling: extract content from response
            if isinstance(response, str):
                output_text = response
            else:
                output_text = response.choices[0].message.content

            if not output_text:
                logging.error("Received empty response from api.")
                return ""

            logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
            return output_text

        except Exception as e:
            if isinstance(e, openai.APIError):  # 针对所有 openai.APIError 进行重试
                logging.warning(f"OpenAI API error: {e}, attempt {attempt + 1} of {retries}. Retrying in {delay} seconds...")
            elif isinstance(e, Exception): # catch other types of exceptions
                logging.error(f"An error occurred (attempt {attempt+1}): {e}")

            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                logging.error(f"Request failed after {retries} attempts.")
                return ""  # Return empty string on error after retries

async def generate_completion_from_local_llm(llm_model_name: str, input_prompt: str,
                                             number_of_tokens_to_generate: int = 100, temperature: float = 0.7,
                                             grammar_file_string: str = None):
    """Generates text completion using a locally loaded LLM model."""
    logging.info(
        f"Starting text completion using model: '{llm_model_name}' for input prompt: '{input_prompt}'")
    llm = load_model(llm_model_name)
    prompt_tokens = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max_tokens = min(number_of_tokens_to_generate,
                              LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for LLM. Chunking the input.")
        chunks = chunk_text(input_prompt, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION, llm_model_name)
        results = []
        for chunk in chunks:
            try:
                output = llm(
                    prompt=chunk,
                    max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION,
                    temperature=temperature,
                )
                results.append(output['choices'][0]['text'])
                logging.info(f"Chunk processed. Output tokens: {output['usage']['completion_tokens']:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        grammar_file_string_lower = grammar_file_string.lower() if grammar_file_string else ""
        if grammar_file_string_lower:
            list_of_grammar_files = glob.glob("./grammar_files/*.gbnf")
            matching_grammar_files = [x for x in list_of_grammar_files if
                                      grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]]
            if len(matching_grammar_files) == 0:
                logging.error(f"No grammar file found matching: {grammar_file_string}")
                raise FileNotFoundError
            grammar_file_path = max(matching_grammar_files, key=os.path.getmtime)
            logging.info(f"Loading selected grammar file: '{grammar_file_path}'")
            llama_grammar = LlamaGrammar.from_file(grammar_file_path)
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature,
                grammar=llama_grammar
            )
        else:
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature
            )
        generated_text = output['choices'][0]['text']
        if grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        finish_reason = str(output['choices'][0]['finish_reason'])
        llm_model_usage_json = json.dumps(output['usage'])
        logging.info(
            f"Completed text completion in {output['usage']['total_time']:.2f} seconds. Beginning of generated text: \n'{generated_text[:150]}'...")
        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "llm_model_usage_json": llm_model_usage_json
        }

# Image Processing Functions
def preprocess_image(image):
    """Preprocesses an image for OCR using thresholding and dilation."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    return Image.fromarray(gray)

def convert_pdf_to_images(input_pdf_file_path: str, max_pages: int = 0, skip_first_n_pages: int = 0,
                          temp_dir: str = None) -> List[Image.Image]:
    """
    Converts a PDF file to a list of PIL Image objects.

    Args:
        input_pdf_file_path (str): The path to the input PDF file.
        max_pages (int, optional): The maximum number of pages to convert. If 0, all pages are converted. Defaults to 0.
        skip_first_n_pages (int, optional): The number of pages to skip from the beginning of the PDF. Defaults to 0.
        temp_dir (str, optional): The path to the temporary directory to store images.
                                If None, it defaults to the value set in the .env file's TEMP_IMAGE_DIR or ".temp_pdf_images".
                                Defaults to None.

    Returns:
        List[Image.Image]: A list of PIL Image objects representing the converted pages.
    """
    logging.info(f"Processing PDF file {input_pdf_file_path}")
    base_name = os.path.splitext(os.path.basename(input_pdf_file_path))[0]
    if max_pages == 0:
        last_page = None  # Convert all pages
        logging.info("Converting all pages to images...")
    else:
        last_page = skip_first_n_pages + max_pages
        logging.info(f"Converting pages {skip_first_n_pages + 1} to {last_page}")

    first_page = skip_first_n_pages + 1  # pdf2image uses 1-based indexing

    temp_dir = TEMP_IMAGE_DIR
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the default temp directory exists

    existing_image_files = glob.glob(os.path.join(temp_dir, f"{base_name}-*.png"))

    if existing_image_files:
        logging.info(f"Found existing images in temporary directory. Skipping PDF conversion.")
        images = []
        if max_pages == 0:
            expected_pages = list(range(first_page, 10000))  # Assume a large number if max_pages is 0
        else:
            expected_pages = list(range(first_page, first_page + max_pages))
        try:
            # Try to find at least one matching image file
            matching_file = next(file for file in existing_image_files)
            images.append(Image.open(matching_file))

            return images
        except StopIteration:
            logging.warning(f"Could not find image")
        except Exception as e:
            logging.error(f"Could not load image: Error: {e}")

    logging.info("No existing images found in temporary directory, proceeding with PDF conversion...")

    images = convert_from_path(
        input_pdf_file_path,
        first_page=first_page,
        last_page=last_page,
        output_folder=temp_dir,
        paths_only=True,
        fmt='png',
        output_file=f"{base_name}-"  # Set output_file to ensure consistent naming
    )

    renamed_images = []
    for image_path in images:
        renamed_images.append(Image.open(image_path))

    logging.info(f"Converted {len(renamed_images)} pages from PDF file to images. Temporary files saved in: {temp_dir}")
    return renamed_images

def ocr_image(image):
    """Performs OCR on a single image using Tesseract."""
    preprocessed_image = preprocess_image(image)
    return pytesseract.image_to_string(preprocessed_image)

async def process_chunk(chunk: str, prev_context: str, chunk_index: int, total_chunks: int,
                        reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool) -> Tuple[str, str]:
    """Processes a single chunk of text, including OCR correction and optional markdown formatting."""
    logging.info(f"Processing chunk {chunk_index + 1}/{total_chunks} (length: {len(chunk):,} characters)")

    # Step 1: OCR Correction
    ocr_correction_prompt = f"""Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and common sense to correct errors
   - Only fix clear errors, don't alter the content unnecessarily
   - Do not add extra periods or any unnecessary punctuation

2. Maintain original structure:
   - Keep all headings and subheadings intact

3. Preserve original content:
   - Keep all important information from the
      - Do not add any new information not present in the original text
   - Remove unnecessary line breaks within sentences or paragraphs
   - Maintain paragraph breaks

4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context
   - Handle text that starts or ends mid-sentence appropriately

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.

Previous context:
{prev_context[-500:]}

Current chunk to process:
{chunk}

Corrected text:
"""

    ocr_corrected_chunk = await generate_completion(ocr_correction_prompt, max_tokens=len(chunk) + 500)

    processed_chunk = ocr_corrected_chunk

    # Step 2: Markdown Formatting (if requested)
    if reformat_as_markdown:
        markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
   - Ensure each heading is on its own line
   - Add a blank line before and after each heading
2. Maintain the original paragraph structure. Remove all breaks within a word that should be a single word (for example, "cor- rect" should be "correct").
3. Format lists properly (unordered or ordered) if they exist in the original text
4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate, based on the original formatting
5. Preserve all original content and meaning
6. Do not add any extra punctuation or modify the existing punctuation
7. Remove any spuriously inserted introductory text such as "Here is the corrected text:" that may have been added by the LLM and which is obviously not part of the original text.
8. Remove any obviously duplicated content that appears to have been accidentally included twice. Follow these strict guidelines:
   - Remove only exact or near-exact repeated paragraphs or sections within the main chunk.
   - Consider the context (before and after the main chunk) to identify duplicates that span chunk boundaries.
   - Do not remove content that is simply similar but conveys different information.
   - Preserve all unique content, even if it seems redundant.
   - Ensure the text flows smoothly after removal.
   - Do not add any new content or explanations.
   - If no obvious duplicates are found, return the main chunk unchanged.
9. {"Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly, e.g., as blockquotes." if not suppress_headers_and_page_numbers else "Carefully remove headers, footers, and page numbers while preserving all other content."}

Text to reformat:

{ocr_corrected_chunk}

Reformatted markdown:
"""
        processed_chunk = await generate_completion(markdown_prompt, max_tokens=len(ocr_corrected_chunk) + 500)
    new_context = processed_chunk[-1000:]  # Use the last 1000 characters as context for the next chunk
    logging.info(f"Chunk {chunk_index + 1}/{total_chunks} processed. Output length: {len(processed_chunk):,} characters")
    return processed_chunk, new_context

async def process_chunks(chunks: List[str], reformat_as_markdown: bool,
                         suppress_headers_and_page_numbers: bool) -> List[str]:
    """Processes multiple chunks of text, handling concurrency for API-based LLMs and sequential processing for local LLMs."""
    total_chunks = len(chunks)

    async def process_chunk_with_context(chunk: str, prev_context: str, index: int) -> Tuple[int, str, str]:
        """Helper function to process a chunk and return its index, processed content, and new context."""
        processed_chunk, new_context = await process_chunk(chunk, prev_context, index, total_chunks,
                                                            reformat_as_markdown, suppress_headers_and_page_numbers)
        return index, processed_chunk, new_context

    if USE_LOCAL_LLM:
        logging.info("Using local LLM. Processing chunks sequentially...")
        context = ""
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk, context = await process_chunk(chunk, context, i, total_chunks, reformat_as_markdown,
                                                            suppress_headers_and_page_numbers)
            processed_chunks.append(processed_chunk)
    elif not ASYNC_API_REQUESTS:  # Handle synchronous API requests
        logging.info("Using API-based LLM with synchronous requests. Processing chunks sequentially...")
        context = ""
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk, context = await process_chunk(chunk, context, i, total_chunks, reformat_as_markdown,
                                                            suppress_headers_and_page_numbers)
            processed_chunks.append(processed_chunk)
    else:
        logging.info("Using API-based LLM. Processing chunks concurrently while maintaining order...")
        tasks = [process_chunk_with_context(chunk, "", i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        # Sort results by index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        processed_chunks = [chunk for _, chunk, _ in sorted_results]
    logging.info(f"All {total_chunks} chunks processed successfully")
    return processed_chunks

async def process_document(list_of_extracted_text_strings: List[str], reformat_as_markdown: bool = True,
                           suppress_headers_and_page_numbers: bool = True) -> str:
    """Processes the entire document, including chunking, processing chunks, and combining the results."""
    logging.info(f"Starting document processing. Total pages: {len(list_of_extracted_text_strings):,}")
    full_text = "\n\n".join(list_of_extracted_text_strings)
    logging.info(f"Size of full text before processing: {len(full_text):,} characters")
    chunk_size, overlap = 8000, 10
    # Improved chunking logic
    paragraphs = re.split(r'\n\s*\n', full_text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_chunk_length + paragraph_length <= chunk_size:
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_length
        else:
            # If adding the whole paragraph exceeds the chunk size,
            # we need to split the paragraph into sentences
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = []
            current_chunk_length = 0
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_chunk_length + sentence_length <= chunk_size:
                    current_chunk.append(sentence)
                    current_chunk_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_length = sentence_length
    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])
    # Add overlap between chunks
    for i in range(1, len(chunks)):
        overlap_text = chunks[i - 1].split()[-overlap:]
        chunks[i] = " ".join(overlap_text) + " " + chunks[i]
    logging.info(f"Document split into {len(chunks):,} chunks. Chunk size: {chunk_size:,}, Overlap: {overlap:,}")
    processed_chunks = await process_chunks(chunks, reformat_as_markdown, suppress_headers_and_page_numbers)
    final_text = "".join(processed_chunks)
    logging.info(f"Size of text after combining chunks: {len(final_text):,} characters")
    logging.info(f"Document processing complete. Final text length: {len(final_text):,} characters")
    return final_text

def remove_corrected_text_header(text):
    """Removes common header artifacts that might be inserted by the LLM during processing."""
    return text.replace("# Corrected text\n", "").replace("# Corrected text:", "").replace("\nCorrected text",
                                                                                        "").replace(
        "Corrected text:", "")

async def assess_output_quality(original_text, processed_text):
    """Assesses the quality of the processed text compared to the original text."""
    max_chars = 8193  # Limit to avoid exceeding token limits
    available_chars_per_text = max_chars // 2  # Split equally between original and processed

    original_sample = original_text[:available_chars_per_text]
    processed_sample = processed_text[:available_chars_per_text]

    prompt = f"""Compare the following samples of original OCR text with the processed output and assess the quality of the processing. Consider the following factors:
1. Accuracy of error correction
2. Improvement in readability
3. Preservation of original content and meaning
4. Appropriate use of markdown formatting (if applicable)
5. Removal of hallucinations or irrelevant content

Original text sample:
```
{original_sample}
```

Processed text sample:
```
{processed_sample}
```

Provide a quality score between 0 and 100, where 100 is perfect processing. Also provide a brief explanation of your assessment.

Your response should be in the following format:
SCORE: [Your score]
EXPLANATION: [Your explanation]
"""

    # Calculate max_tokens dynamically
    prompt_tokens = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    max_tokens = OPENAI_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER

    # Ensure max_tokens is positive
    max_tokens = max(1, max_tokens)  # Use at least 1 token

    response = await generate_completion(prompt, max_tokens=max_tokens)

    try:
        lines = response.strip().split('\n')
        score_line = next(line for line in lines if line.startswith('SCORE:'))
        score = int(score_line.split(':')[1].strip())
        explanation = '\n'.join(line for line in lines if line.startswith('EXPLANATION:')).replace('EXPLANATION:',
                                                                                                  '').strip()
        logging.info(f"Quality assessment: Score {score}/100")
        logging.info(f"Explanation: {explanation}")
        return score, explanation
    except Exception as e:
        logging.error(f"Error parsing quality assessment response: {e}")
        logging.error(f"Raw response: {response}")
        return None, None

async def main():
    """Main function to run the PDF processing pipeline."""
    try:
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        max_test_pages = 0  # Set to 0 to process all pages, or a positive integer to limit the number of pages
        skip_first_n_pages = 0  # Set to the number of pages to skip from the beginning
        reformat_as_markdown = True  # Set to True to format the output as markdown
        suppress_headers_and_page_numbers = True # Set to True to remove headers, footers and page numbers
        base_name = os.path.splitext(input_pdf_file_path)[0]
        output_extension = '.md' if reformat_as_markdown else '.txt'
        # Download the model if using local LLM
        if USE_LOCAL_LLM:
            _, download_status = await download_models()
            logging.info(f"Model download status: {download_status}")
            logging.info(f"Using Local LLM with Model: {DEFAULT_LOCAL_MODEL_NAME}")
        else:
            logging.info(f"Using API for completions: {API_PROVIDER}")
            logging.info(f"Using OpenAI model for embeddings: {OPENAI_EMBEDDING_MODEL}")

        raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
        llm_corrected_output_file_path = base_name + '_llm_corrected' + output_extension

        if os.path.exists(raw_ocr_output_file_path):
            logging.info(
                f"Found existing raw OCR output file: {raw_ocr_output_file_path}. Skipping PDF conversion and OCR.")
            with open(raw_ocr_output_file_path, "r", encoding="utf-8") as f:
                raw_ocr_output = f.read()
            list_of_extracted_text_strings = [raw_ocr_output]  # Wrap string in a list for compatibility
            logging.info(f"Raw OCR output read from: {raw_ocr_output_file_path}")
        else:
            list_of_scanned_images = convert_pdf_to_images(input_pdf_file_path, max_test_pages,
                                                            skip_first_n_pages)  # Use the default temp dir
            logging.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
            logging.info("Extracting text from converted pages...")
            with ThreadPoolExecutor() as executor:
                list_of_extracted_text_strings = list(executor.map(ocr_image, list_of_scanned_images))
            logging.info("Done extracting text from converted pages.")
            raw_ocr_output = "\n".join(list_of_extracted_text_strings)
            with open(raw_ocr_output_file_path, "w", encoding="utf-8") as f:
                f.write(raw_ocr_output)
            logging.info(f"Raw OCR output written to: {raw_ocr_output_file_path}")

        logging.info("Processing document...")
        final_text = await process_document(list_of_extracted_text_strings, reformat_as_markdown,
                                            suppress_headers_and_page_numbers)
        cleaned_text = remove_corrected_text_header(final_text)

        # Save the LLM corrected output
        with open(llm_corrected_output_file_path, 'w', encoding="utf-8") as f:
            f.write(cleaned_text)
        logging.info(f"LLM Corrected text written to: {llm_corrected_output_file_path}")

        if final_text:
            logging.info(f"First 500 characters of LLM corrected processed text:\n{final_text[:500]}...")
        else:
            logging.warning("final_text is empty or not defined.")

        logging.info(f"Done processing {input_pdf_file_path}.")
        logging.info("\nSee output files:")
        logging.info(f" Raw OCR: {raw_ocr_output_file_path}")
        logging.info(f" LLM Corrected: {llm_corrected_output_file_path}")

        # Perform a final quality check
        quality_score, explanation = await assess_output_quality(raw_ocr_output, final_text)
        if quality_score is not None:
            logging.info(f"Final quality score: {quality_score}/100")
            logging.info(f"Explanation: {explanation}")
        else:
            logging.warning("Unable to determine final quality score.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())