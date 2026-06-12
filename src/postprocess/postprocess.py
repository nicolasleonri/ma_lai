"""
postprocess.py

This script performs postprocessing on raw OCR-extracted newspaper text using LLMs via Ollama.
It converts unstructured text into structured CSV files, extracting articles, headlines, subheadlines,
authors, and content for each image-configuration pair.

Supports multithreaded processing and multiple model runs.

Author: @nicolasleonri (GitHub)
License: GPL
"""

from utils_postprocessing import parse_ocr_results, extract_filename_and_config, extract_code_block, log_processing_info, parse_image_results, load_shared_inputs, image_to_base64png, log_processing_info_olmo
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
from qwen_vl_utils import process_vision_info
from multiprocessing import Pool, cpu_count
from ollama import chat, ChatResponse
from PIL import Image, ImageDraw
import multiprocessing as mp
from threading import Lock
from csv import DictReader
from io import StringIO
from queue import Queue
import multiprocessing
from tqdm import tqdm
import typing as T
import threading
import unittest
import argparse
import torch
import time
import json
import sys
import csv
import re
import os


class VLMProcessor:
    """Wrapper around VLM models for structured postprocessing of image content."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name (str): Name of the VLM model to use.
            device (str): Device to run the model on ('auto', 'cuda', 'cpu', 'mps').
        """
        self.model_name = model_name
        self.processor = None
        self.tokenizer = None
        self.model = None
        self._model_lock = Lock()  # For thread safety
        self._load_model()

    def _load_model(self):
        """Load the VLM model and processor"""
        try:
            print(f"Loading {self.model_name}")
            if self.model_name == "allenai/olmOCR-7B-0225-preview":
                # Special loading for OLM OCR
                self.model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview",
                                                                             torch_dtype=torch.bfloat16,
                                                                             attn_implementation="flash_attention_2").eval()
                self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
            elif self.model_name == "reducto/RolmOCR":
                # Special loading for RolmOCR
                self.device = torch.device("cuda")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    "reducto/RolmOCR",
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(
                    "reducto/RolmOCR", trust_remote_code=True, use_fast=True, device_map="auto")
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, use_fast=True)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation="flash_attention_2")
                self.model.eval()  # Set to evaluation mode

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, use_fast=True)
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, use_fast=True)

            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def extract_test_results_as_csv(self, image_path: str) -> str:
        """Uses a VLM to extract image content into CSV-formatted string.
        Args:
            image_path (str): Path to the image file.
        Returns:
            str: CSV string in the same format as the LLM version.
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            prompt = """
            You are an expert OCR system. Extract ALL text content from this newspaper image with perfect accuracy.

            CRITICAL REQUIREMENTS:
            - Read every single word, number, date, and punctuation mark visible in the image
            - The text is in SPANISH - preserve all Spanish accents, tildes, and special characters (ñ, á, é, í, ó, ú, ü)
            - Preserve the original text layout and structure (headlines, paragraphs, columns)
            - Maintain proper spacing between words and sentences
            - Include ALL content: headlines, subheadings, body text, captions, advertisements, page numbers, dates
            - Handle multiple columns by reading left-to-right, top-to-bottom within each column
            - Preserve special characters, accents, and non-English text exactly as shown
            - Do NOT skip any text, even if partially obscured or small
            - Do NOT add explanations, interpretations, or markdown formatting
            - Do NOT summarize or paraphrase - extract the exact text as written
            
            Return ONLY the raw extracted text content, preserving the natural reading flow of the newspaper.
            
            WARNING: If you return anything other than raw text (explanations, apologies, formatting, etc.), 
            the entire OCR pipeline will fail and all downstream processing will be corrupted. 
            Your response must contain ONLY the extracted text - nothing else.
            """
            # Process the image with thread safety
            with self._model_lock:
                if self.model_name == "allenai/olmOCR-7B-0225-preview":
                    messages = [
                        {
                            "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image", "image": image},
                                    ],
                        }
                    ]

                    # Apply the chat template and processor
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = {key: value.to(self.device)
                              for (key, value) in inputs.items()}

                    # TODO: Play with these parameters
                    output = self.model.generate(
                        **inputs,
                        temperature=0.1,
                        max_new_tokens=30000,
                        num_return_sequences=1,
                        do_sample=False,
                        # num_beams=3,
                        # early_stopping=True,
                    )

                    # Decode the output
                    prompt_length = inputs["input_ids"].shape[1]
                    new_tokens = output[:, prompt_length:]
                    text_output = self.processor.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )

                    text = text_output[0]

                    try:
                        data = json.loads(text)
                        text = data["natural_text"]
                    except Exception as e:
                        pass

                    output_text = text if text else ""

                elif self.model_name == "reducto/RolmOCR":
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "image", "image": f"file://{image_path}"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=50000, do_sample=False, temperature=0.1)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    output_text = output_text[0] if output_text else ""
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": prompt},
                        ]},
                    ]

                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text], images=[
                                            image], padding=True, return_tensors="pt")
                    inputs = inputs.to(self.model.device)

                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=3500, do_sample=False, temperature=0.1)
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
                        inputs.input_ids, output_ids)]

                    output_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    output_text = output_text[0] if output_text else ""
            return output_text

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return f"Error processing image: {str(e)}"

        finally:
            if image:
                image.close()


class ollama:
    """Wrapper around Ollama-based LLMs for structured postprocessing of OCR content."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name (str): Name of the LLM model to use via Ollama.
        """
        self.model_name = model_name

    def chat_completion(self, prompt: str, question: str) -> str:
        """Sends a chat completion request to Ollama.

        Args:
            prompt (str): System-level instruction.
            question (str): User query containing OCR text.

        Returns:
            str: LLM response content.
        """
        response: ChatResponse = chat(
            model=self.model_name,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": question}],
            options={"temperature": 0.1, "num_ctx": 30000},
        )
        response = response['message']['content']
        return response

    def extract_test_results_as_csv(self, ocr_text: str) -> str:
        """Uses an LLM to extract OCR content into CSV-formatted string.

        Args:
            ocr_text (str): Raw OCR text.

        Returns:
            str: CSV string in markdown code block format.
        """

        prompt = """
        GOAL: Given the image, extract and structure the following information:
        - headline of the article (string or "NA")
        - subheadline of the article (string or "NA")
        - author of the article (string or "NA")
        - content of the article (string)

        IMPORTANT:
        - Focus only on articles that contain meaningful journalistic content.
        - Exclude very short notices such as: date blocks, weather updates, advertisements, or public announcements.
        - Ask yourself: is this content relevant for media or discourse analysis? If not, skip it.
        - If any field is missing or unknown, write "NA".

        RETURN FORMAT:
        Strictly output a valid CSV in the following format:
        "headline";"subheadline";"author";"content"
        "El loco del martillo";"NA";"La Seño María";"Hoy en día, uno pensaría que..."
        "Contento por fin de cuarentena";"Habla Trome";"Ismael Lazo, Vecino de San Luis";"Estoy feliz porque..."

        RULES:
        - Do NOT include explanations, extra text, or commentary.
        - Enclose each field in double quotes.
        - Use semicolons (`;`) as field separators.
        - Do NOT insert semicolons inside fields. If needed, replace them with commas.
        - Each row represents one article. The first row must always be the CSV header.

        CONTEXT:
        You are an expert in analyzing and structuring newspaper content. Extracting accurate information is your professional responsibility. Be precise and thorough. If you make a mistake, the CSV will break and your credibility will suffer.
        """

        response = self.chat_completion(prompt, ocr_text)

        return response

    def read_txt_file(self, file_path: str) -> dict:
        """Reads OCR log file and prompts each result using the LLM.

        Args:
            file_path (str): Path to the OCR log output.

        Returns:
            dict: Dictionary of enriched OCR results keyed by index.
        """
        ocr_results = parse_ocr_results(os.path.join(os.getcwd(), file_path))

        for key, value in ocr_results.items():
            print(
                f"Prompting extracted text from {value['filepath']} using module {value['config']}:")
            start = time.time()
            structured_results = self.extract_test_results_as_json(
                value['text'])
            ocr_results[key]["Output"] = structured_results
            end = time.time()
            time_elapsed = end - start
            print(f"Time nedded: {round(time_elapsed, 5)}")

        return ocr_results


def process_single_result(key: int, value: dict, model_name: str, model_display_name: str, progress_queue: Queue, log_file_path=str, model_input=any) -> dict:
    """Processes one OCR result with a specified LLM model.

    Args:
        key (int): Dictionary key for the OCR result.
        value (dict): OCR result metadata and text.
        model_name (str): LLM model identifier.
        model_display_name (str): Short label for display/logging.
        progress_queue (Queue): Shared queue for reporting progress.

    Returns:
        dict: Result including extracted data, runtime, and status.
    """

    vlm_models = {
        "nanonets/Nanonets-OCR-s": "nanonets",
        "reducto/RolmOCR": "rolmocr",
        "allenai/olmOCR-7B-0225-preview": "olmocr",
    }

    try:
        start = time.time()

        if model_name not in vlm_models:
            ocr_name = str(value['ocr'])
            config_no = str(value['config'])
            processed_filename, _ = extract_filename_and_config(
                value['filepath'])
            structured_results = model_input.extract_test_results_as_csv(
                value['text'])
        elif model_name in vlm_models:
            ocr_name = str(model_display_name)
            processed_filename, config_no = extract_filename_and_config(
                value['filepath'])
            
            if 'cropped_folder_flag' in value:
                filepath_without_suffix = os.path.splitext(value['filepath'])[0]
                image_paths = parse_image_results(filepath_without_suffix)
                file_list = []
                for idx, val in enumerate(image_paths.values()):
                    file_list.append(val['filepath'])
                all_results = []
                for val in file_list:
                    result = model_input.extract_test_results_as_csv(val)
                    all_results.append(result)
                structured_results = " ".join(all_results)
            else:
                structured_results = model_input.extract_test_results_as_csv(
                value['filepath'])

        end = time.time()
        time_elapsed = end - start

        if model_name in vlm_models:
            print("VLM detected, logging to OCR log.")
            log_file_path = "./results/txt/extracted/vlm_results_log.txt"
            log_processing_info_olmo(log_file_path, processed_filename,
                                     config_no, model_display_name, time_elapsed, structured_results)
        else:
            log_processing_info(log_file_path, processed_filename,
                                config_no, ocr_name, model_display_name, time_elapsed)

        result = {
            'key': key,
            'value': value,
            'output': structured_results,
            'time_elapsed': time_elapsed,
            'model_name': model_display_name,
            'success': True,
            'error': None
        }

        try:
            progress_queue.put(result)  # Send progress update
        except:
            pass

        return result

    except Exception as e:
        error_result = {
            'key': key,
            'value': value,
            'output': None,
            'time_elapsed': 0,
            'model_name': model_display_name,
            'success': False,
            'error': str(e)
        }

        progress_queue.put(error_result)
        return error_result


def progress_monitor(progress_queue: Queue, total_tasks: int, results_dict: dict, lock: threading.Lock) -> None:
    """Consumes result objects from the queue and tracks processing progress.

    Args:
        progress_queue (Queue): Queue receiving LLM task results.
        total_tasks (int): Total number of expected results.
        results_dict (dict): Shared dictionary storing all results.
        lock (threading.Lock): Thread lock for safe writes.
    """
    completed = 0

    while completed < total_tasks:
        try:
            result = progress_queue.get(timeout=1)
            completed += 1

            with lock:
                results_dict[result['key']] = result

            # Print progress
            percentage = round(completed / total_tasks * 100, 1)
            if result['success']:
                print(f"✓ Progress: {completed}/{total_tasks} ({percentage}%) - "
                      f"Processed {result['value']['filepath']} with {result['model_name']} "
                      f"in {round(result['time_elapsed'], 2)}s")
            else:
                print(f"✗ Progress: {completed}/{total_tasks} ({percentage}%) - "
                      f"Failed {result['value']['filepath']} with {result['model_name']} "
                      f"- Error: {result['error']}")

            progress_queue.task_done()

        except:
            continue  # Timeout, continue waiting


class Article(T.NamedTuple):
    """Typed representation of a structured newspaper article."""
    headline: str
    subheadline: str
    author: str
    content: str

    @classmethod
    def from_row(cls, row: dict):
        return cls(**{
            key: type_(row[key]) for key, type_ in cls._field_types.items()
        })


def validate_csv(reader: DictReader) -> bool:
    """Validates that a CSV reader produces rows compatible with Article.

    Args:
        reader (DictReader): CSV reader object.

    Returns:
        bool: True if all rows are valid Article entries; False otherwise.
    """
    for row in reader:
        try:
            Article.from_row(row)
        except Exception as e:
            print('type: {} msg: {}'.format(type(e), e))
            return False
    return True


def save_results_to_csv(results_dict: dict, model_display_name: str) -> T.Tuple[int, int]:
    """Saves structured OCR results (CSV-formatted strings) into CSV files with metadata.

    Args:
        results_dict (dict): Dictionary of processed results.
        model_display_name (str): Short model name to include in filenames.

    Returns:
        Tuple[int, int]: (successful saves, errors encountered)
    """
    saved_count = 0
    error_count = 0

    vlm_models = {
        "nanonets/Nanonets-OCR-s": "nanonets",
        "reducto/RolmOCR": "rolmocr",
        "allenai/olmOCR-7B-0225-preview": "olmocr",
    }

    for key, result in results_dict.items():
        try:
            if model_display_name in vlm_models.values():
                print("VLM model detected, skipping CSV.")
                continue

            value = result['value']
            filename = os.path.splitext(os.path.basename(value['filepath']))[0]
            filename = re.sub(r'_config\d+$', '', filename)
            ocr_name = value["ocr"]
            config_no = value['config']
            pathfile = f"{filename}_config{config_no}_{ocr_name}_{model_display_name}"
            file_name_csv = f"./results/csv/extracted/{pathfile}.csv"

            # Extract metadata from filename (e.g., newspaper#date#page_imgXX)
            metadata_parts = filename.split('_')[0].split('#')

            if len(metadata_parts) == 3:
                newspaper_name, publication_date, page_str = metadata_parts
                try:
                    page_number = int(page_str)
                except ValueError:
                    page_number = None
            else:
                newspaper_name = publication_date = None
                page_number = None

            document_metadata = {
                "newspaper_name": newspaper_name,
                "publication_date": publication_date,
                "page_number": page_number
            }
            data = extract_code_block(result['output'], language_hint="csv")
            f = StringIO(data)
            reader = csv.reader(f, delimiter=';', quotechar='"')

            with open(file_name_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(
                    f,
                    delimiter=';',
                    quotechar='"',
                    quoting=csv.QUOTE_ALL  # Force quotation around every field
                )

                # Extended header
                header = ["headline", "subheadline", "author", "content",
                          "newspaper_name", "publication_date", "page_number"]
                writer.writerow(header)

                # Write data rows
                for row in reader:
                    if row == ["headline", "subheadline", "author", "content"]:
                        continue
                    if len(row) == 4:
                        full_row = row + [
                            document_metadata["newspaper_name"],
                            document_metadata["publication_date"],
                            document_metadata["page_number"]
                        ]
                        writer.writerow(full_row)

            print(f"✓ Results saved: {file_name_csv}")
            saved_count += 1
        except Exception as e:
            print(f"✗ Error saving for {value['filepath']}: {e}")

            if model_display_name == "allenai/olmOCR-7B-0225-preview":
                print("OLM OCR model detected, skipping CSV.")
                continue

            error_count += 1

            try:
                # Use the same filename construction as above
                filename = os.path.splitext(
                    os.path.basename(value['filepath']))[0]
                ocr_name = value["ocr"]
                config_no = value['config']
                pathfile = f"{filename}_config{config_no}_{ocr_name}_{model_display_name}"
                file_name_csv = f"./results/csv/extracted/{pathfile}.csv"

                with open(file_name_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(
                        f,
                        delimiter=';',
                        quotechar='"',
                        quoting=csv.QUOTE_ALL
                    )
                    header = ["headline", "subheadline", "author", "content",
                              "newspaper_name", "publication_date", "page_number"]
                    writer.writerow(header)
                print(f"✓ Empty CSV saved due to error: {file_name_csv}")
            except Exception as inner_e:
                print(f"✗ Failed to save empty CSV on error: {inner_e}")
            continue

    return saved_count, error_count


def save_results_to_json(results_dict: dict, model_display_name: str) -> T.Tuple[int, int]:
    """Saves structured OCR results (JSON) into JSON files.

    Args:
        results_dict (dict): Dictionary of processed results.
        model_display_name (str): Short model name to include in filenames.

    Returns:
        Tuple[int, int]: (successful saves, errors encountered)
    """
    saved_count = 0
    error_count = 0

    for key, result in results_dict.items():
        if not result['success']:
            error_count += 1
            print(f"Skipping failed result for {result['value']['filepath']}")
            continue

        try:
            value = result['value']
            filename = os.path.splitext(os.path.basename(value['filepath']))[0]
            config_name = value["config"]
            pathfile = f"{filename}_{config_name}_{model_display_name}"
            file_name_json = f"./results/txt/extracted/{pathfile}.json"

            metadata_parts = filename.split('_')[0].split('#')

            if len(metadata_parts) == 3:
                newspaper_name, publication_date, page_str = metadata_parts
                try:
                    page_number = int(page_str)
                except ValueError:
                    page_number = None
            else:
                newspaper_name = publication_date = None
                page_number = None

            document_metadata = {
                "newspaper_name": newspaper_name,
                "publication_date": publication_date,
                "page_number": page_number
            }

            try:
                json_object = json.loads(result['output'])
            except (ValueError, json.JSONDecodeError) as e:
                print(f"JSON Parse ERROR for {value['filepath']}: {e}")
                error_count += 1
                continue

            # Add document_metadata key if not already present
            if "metadata" not in json_object:
                json_object["document_metadata"] = document_metadata

            # Save to file
            with open(file_name_json, 'w', encoding='utf-8') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)

            print(f"✓ Results saved: {file_name_json}")
            saved_count += 1

        except Exception as e:
            print(f"Save ERROR for {result['value']['filepath']}: {e}")
            error_count += 1

    return saved_count, error_count


def process_model_multithreaded(model_name: str, model_display_name: str, ocr_results: dict, img_results: dict, max_workers: int = 3, log_file_path: str = "None") -> dict:
    """Processes all OCR results using a specific LLM with multithreading.

    Args:
        model_name (str): Internal LLM model identifier for Ollama.
        model_display_name (str): Friendly name for filenames/logs.
        ocr_results (dict): Parsed OCR results.
        max_workers (int, optional): Thread count.

    Returns:
        dict: Result dictionary of processed entries.
    """
    vlm_models = {
        "nanonets/Nanonets-OCR-s": "nanonets",
        "reducto/RolmOCR": "rolmocr",
        "allenai/olmOCR-7B-0225-preview": "olmocr",
    }

    print(f"\n{'='*60}")
    print(f"Starting multithreaded processing for model: {model_display_name}")
    if model_name in vlm_models:
        print(f"Total Images to process: {len(img_results)}")
    else:
        print(f"Total OCR results to process: {len(ocr_results)}")
    # print(f"Total Images to process: {len(img_results)}")
    print(f"Using {max_workers} worker threads")
    print(f"{'='*60}")

    # Determine actual input based on model type
    if model_name not in vlm_models:
        actual_input = ocr_results
    else:
        actual_input = img_results

    # Create progress tracking
    progress_queue = Queue()
    results_dict = {}
    lock = threading.Lock()

    # Start progress monitor thread with the correct task count
    progress_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_queue, len(actual_input), results_dict, lock)
    )
    progress_thread.start()

    # Process all results in parallel
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        if model_name not in vlm_models:
            llm = ollama(model_name)  # Create LLM instance for this thread
            # Submit all tasks
            for key, value in ocr_results.items():
                future = executor.submit(
                    process_single_result,
                    key, value, model_name, model_display_name, progress_queue, log_file_path, llm
                )
                futures.append(future)
        elif model_name in vlm_models:
            vlm = VLMProcessor(model_name)  # Load once here
            for key, value in img_results.items():
                future = executor.submit(
                    process_single_result,
                    key, value, model_name, model_display_name, progress_queue, log_file_path, vlm
                )
                futures.append(future)

        # Wait for all tasks to complete
        completed_tasks = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_tasks += 1
            except Exception as e:
                completed_tasks += 1
                print(f"Task execution error: {e}")

    # Wait for progress monitor to finish
    progress_thread.join()

    processing_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Completed processing for {model_display_name}")
    print(f"Total processing time: {round(processing_time, 2)} seconds")
    print(
        f"Average time per result: {round(processing_time / len(ocr_results), 2)} seconds")

    print(f"\nSaving results to CSV files...")
    saved_count, error_count = save_results_to_csv(
        results_dict, model_display_name)

    print(f"✓ Successfully saved: {saved_count} files")
    print(f"✗ Errors encountered: {error_count} files")
    print(f"{'='*60}")

    return results_dict


def initializer_worker(model_name, model_display_name, log_file_path):
    global model_instance, model_name_global, model_display_name_global, log_file_path_global
    model_name_global = model_name
    model_display_name_global = model_display_name
    log_file_path_global = log_file_path

    vlm_models = {
        "nanonets/Nanonets-OCR-s",
        "reducto/RolmOCR",
        "allenai/olmOCR-7B-0225-preview"
    }

    if model_name in vlm_models:
        print(f"[{model_display_name}] Initializing VLMProcessor")
        model_instance = VLMProcessor(model_name)
    else:
        print(f"[{model_display_name}] Initializing Ollama model")
        model_instance = ollama(model_name)


def process_worker(args):
    key, value = args
    return process_single_result(
        key, value,
        model_name_global,
        model_display_name_global,
        None,
        log_file_path_global,
        model_instance
    )


def process_model_multiprocessed(model_name: str, model_display_name: str,
                                 ocr_results: dict, img_results: dict,
                                 num_processes: int = None,
                                 log_file_path: str = "None",
                                 cropped_folder_flag: bool = False) -> dict:
    
    vlm_models = {
        "nanonets/Nanonets-OCR-s": "nanonets",
        "reducto/RolmOCR": "rolmocr",
        "allenai/olmOCR-7B-0225-preview": "olmocr",
    }

    print(f"\n{'='*60}")
    print(f"Starting **multiprocessing** for model: {model_display_name}")
    actual_input = img_results if model_name in vlm_models else ocr_results
    print(f"Total items to process: {len(actual_input)}")
    print(f"Using {num_processes or cpu_count()} worker processes")
    print(f"{'='*60}")

    with Pool(
        processes=num_processes or cpu_count(),
        initializer=initializer_worker,
        initargs=(model_name, model_display_name, log_file_path)
    ) as pool:
        keys = list(actual_input.keys())
        values = list(actual_input.values())

        if cropped_folder_flag:
            for value in values:
                value['cropped_folder_flag'] = cropped_folder_flag

        jobs = zip(keys, values)
        results = []
        for res in tqdm(pool.imap_unordered(process_worker, jobs), total=len(actual_input), desc=f"{model_display_name} Progress"):
            results.append(res)

    results_dict = {res['key']: res for res in results}

    # TODO: Add saving results to CSV or JSON
    # print(f"\nSaving results to CSV files...")
    # saved_count, error_count = save_results_to_csv(
    #     results_dict, model_display_name)
        
    print(f"✓ Completed multiprocessing for {model_display_name}")
    return results_dict


def print_help() -> None:
    """Display usage instructions for the LLM postprocessing script."""
    help_text = """
    LLM/VLM Postprocessing Script — Newspaper OCR Pipeline
    ======================================================

    This script postprocesses OCR-extracted text using LLMs or VLMs to generate structured outputs (CSV or JSON).
    It extracts article metadata: headline, subheadline, author, and content.

    USAGE EXAMPLES:
        python postprocess.py --employ-llms              Run LLM postprocessing on OCR logs (default OCR log file)
        python postprocess.py --employ-llms --process-vlms-outputs
                                                         Run LLMs using VLM-generated output as input
        python postprocess.py --employ-llms --cropped_folder
                                                         Run LLMs on OCR logs from cropped images
        python postprocess.py --employ-vlms              Run VLMs (OLM OCR, RolmOCR, etc.) on preprocessed images
        python postprocess.py --multi --employ-vlms      Enable multiprocessing (multi-GPU or parallel CPU) for VLMs
        python postprocess.py --test                     Run all unit tests
        python postprocess.py --help or -h               Show this help message

    INPUT FILES:
        ./results/txt/extracted/ocr_results_log.txt       ← Default OCR logs from text-based OCR
        ./results/txt/extracted/vlm_results_log.txt       ← Logs from VLM-based OCR (e.g., OLM OCR)
        ./results/txt/extracted/ocr_cropped_results_log.txt ← Logs from OCR on cropped article images
        ./results/images/preprocessed/                    ← Preprocessed newspaper images

    OUTPUT FILES:
        ./results/csv/extracted/                          ← CSV files with extracted article metadata
        ./results/txt/extracted/                          ← Additional output logs (OCR/VLM structured logs)

    FLAGS:
        --employ-llms               Use LLMs for structured metadata extraction from OCR output
        --employ-vlms               Use vision-language models (VLMs) for OCR
        --multi                     Enable multiprocessing mode for faster VLM inference
        --process-vlms-outputs      Use VLM-generated logs as input for LLMs
        --cropped_folder            Use cropped OCR logs for LLM postprocessing
        --test                      Run unit tests
        --help, -h                  Show this help message and exit

    REQUIREMENTS:
        - Ollama (for LLMs) must be installed and running locally
        - Hugging Face models must be pulled beforehand if not using cached versions

    SUPPORTED MODELS:
        VLMs:
            - allenai/olmOCR-7B-0225-preview
            - reducto/RolmOCR
            - nanonets/Nanonets-OCR-s

        LLMs via Ollama:
            - phi4
            - qwen3
            - mistral-nemo
            - deepseek-r1
            - gemma3

    """
    print(help_text.strip())


class TestPostprocessing(unittest.TestCase):
    def test_csv_mock(self):
        """Test code block extraction from mock CSV response."""
        mock_response = '''```csv
"headline";"subheadline";"author";"content"
"Test Title";"Test Subtitle";"John Doe";"This is a test article."
```'''
        csv_block = extract_code_block(mock_response, language_hint="csv")
        reader = csv.reader(StringIO(csv_block), delimiter=';', quotechar='"')
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], "Test Title")
        self.assertEqual(rows[1][2], "John Doe")

    def test_single_ollama_model(self):
        """Run a real or mock LLM call using one model and minimal input."""
        from postprocess import ollama

        sample_text = (
            "1. Headline: 'Breaking News'\n"
            "   Subheadline: 'This is a subhead'\n"
            "   Author: John Doe\n"
            "   Content: Today something very important happened in the city."
        )

        try:
            # Replace with any model available locally
            model = ollama(model_name="phi4:14b")
            response = model.extract_test_results_as_csv(sample_text)
        except Exception as e:
            self.fail(f"Ollama test failed: {e}")


def employ_vlms(args) -> None:
    global cropped_folder_flag 
    cropped_folder_flag = False

    models = {
        "allenai/olmOCR-7B-0225-preview": "olmocr",
        "reducto/RolmOCR": "rolmocr",
        "nanonets/Nanonets-OCR-s": "nanonets",
    }
    print("\nLoading Image list...")
    images_directory = "./results/images/preprocessed"
    img_results = parse_image_results(images_directory)

    # TODO: Read cropped OCR results if --cropped_folder is set
    if args.cropped_folder:
        cropped_folder_flag = True

    ocr_results = parse_ocr_results(os.path.join(
        os.getcwd(), "./results/txt/extracted/ocr_results_log.txt"))
    if os.path.exists("./results/txt/extracted/vlm_results_log.txt"):
        os.remove("./results/txt/extracted/vlm_results_log.txt")
    log_file_path = os.path.join(log_dir, "postprocess_vlms.txt")

    main(models, img_results, ocr_results, log_file_path, args)


def employ_llms(args) -> None:
    global cropped_folder_flag 
    cropped_folder_flag = False

    models = {
        # "phi4:14b": "phi4",
        # "qwen3:32b": "qwen3",
        # "mistral-nemo:12b": "mistral-nemo",
        "deepseek-r1:32b": "deepseek-r1",
        # "gemma3:27b": "gemma3",
        # "llama3.3:70b": "llama3.3", # 43GB so not used!
    }

    if args.process_vlms_outputs:
        ocr_results, img_results = load_shared_inputs(
            "./results/txt/extracted/vlm_results_log.txt")
        log_file_path = os.path.join(log_dir, "postprocess_vlms.txt")
    elif args.cropped_folder:
        ocr_results, img_results = load_shared_inputs(
            "./results/txt/extracted/ocr_cropped_results_log.txt")
        log_file_path = os.path.join(log_dir, "postprocess_cropped.txt")
    else:
        ocr_results, img_results = load_shared_inputs(
            "./results/txt/extracted/ocr_results_log.txt")
        log_file_path = os.path.join(log_dir, "postprocess.txt")

    main(models, img_results, ocr_results, log_file_path, args)


def main(models, img_results, ocr_results, log_file_path, args) -> None:
    print("Starting multithreaded LLM postprocessing...")
    print(f"Models to process: {list(models.values())}")

    max_threads = mp.cpu_count()
    os.makedirs("./results/csv/extracted/", exist_ok=True)

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # if cropped_folder_flag is None:
    #     cropped_folder_flag = False

    # Process each model
    all_results = {}
    total_start_time = time.time()

    for model_name, model_display_name in models.items():
        try:
            if args.multi:
                # print(cropped_folder_flag)
                model_results = process_model_multiprocessed(
                    model_name,
                    model_display_name,
                    ocr_results,
                    img_results,
                    num_processes=3,  # or None for auto
                    log_file_path=log_file_path,
                    cropped_folder_flag=cropped_folder_flag
                )
            else:
                model_results = process_model_multithreaded(
                    model_name,
                    model_display_name,
                    ocr_results,
                    img_results,
                    max_workers=max_threads,
                    log_file_path=log_file_path
                )
            all_results[model_display_name] = model_results

        except Exception as e:
            print(f"Error processing model {model_display_name}: {e}")

    total_time = time.time() - total_start_time

    print(f"{'='*80}")
    print("FINAL SUMMARY".center(80))
    print(f"{'='*80}")
    print(f"Total execution time: {round(total_time, 2)} seconds")
    print(f"Models processed: {len(all_results)}")
    print(f"Total OCR results per model: {len(ocr_results)}")
    print(f"Total processing tasks: {len(models) * len(ocr_results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = None

    model_instance = None
    model_name_global = None
    model_display_name_global = None
    log_file_path_global = None

    parser = argparse.ArgumentParser(
        description="Run VLM processing", add_help=False)
    parser.add_argument("--multi", action="store_true",
                        help="Enable multiprocessing mode (parallel execution)")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--help", "-h", action="store_true",
                        help="Show help message and exit")
    parser.add_argument("--employ-vlms", action="store_true",
                        help="Employ VLMs for text processing")
    parser.add_argument("--process-vlms-outputs",
                        action="store_true", help="Process VLMs outputs")
    parser.add_argument("--employ-llms", action="store_true",
                        help="Employ LLMs for text processing")
    parser.add_argument("--cropped_folder", action="store_true",
                        help="Emply VLMs for image processing")

    args = parser.parse_args()

    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=True)

    if args.help:
        print_help()
        sys.exit(0)

    if args.employ_vlms:
        employ_vlms(args)
        sys.exit(0)

    if args.employ_llms:
        employ_llms(args)
        sys.exit(0)
