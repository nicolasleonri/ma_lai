#!/usr/bin/env python3

import gc
import os
import re
import sys
import csv
import torch
import math
import time
import json
import tiktoken
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from io import StringIO
from pathlib import Path
from transformers import AutoTokenizer
from typing import Union, List, Dict, Any, Optional, Tuple
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from vllm.distributed.parallel_state import destroy_model_parallel

def check_cpu() -> None:
   total, used, free = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
   print("RAM: ", used, " (used)", free, " (free)")

def check_gpu() -> None:
   result = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
      stdout=subprocess.PIPE, text=True, check=False
   )
   print("GPU Memory:", result.stdout.strip())

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract structured information from text files using LLM", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", "--input_folder", type=str, required=True, help="Path to folder containing input text files"
    )
    parser.add_argument("-tn", "--type_of_nvidia_card", type=str, required=True,
        choices=["H100", "A5000", "RTX2080", "A100", "V100"], help="Type of NVIDIA GPU card"
    )
    parser.add_argument("-aram", "--available_ram", type=int, required=True, help="Available system RAM in GB"
    )
    parser.add_argument("-ngpus", "--num_gpus", type=int, required=True, help="Number of GPUs available for processing")
    return parser.parse_args()

def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4

def validate_arguments(args) -> Dict[str, Any]:
    config = {}
    errors = []
    
    if not os.path.exists(args.input_folder):
        errors.append(f"Input folder does not exist: {args.input_folder}")
    else:
        txt_files = list(Path(args.input_folder).rglob("*.txt"))
        if len(txt_files) == 0:
            errors.append(f"No .txt files found in input folder: {args.input_folder}")
        config["txt_files_count"] = len(txt_files)
        config["input_folder"] = args.input_folder
    
    if args.available_ram <= 0:
        errors.append("Available RAM must be positive")
    if args.num_gpus <= 0:
        errors.append("Number of GPUs must be positive")
    
    config["gpu_type"] = args.type_of_nvidia_card
    config["system_ram_gb"] = args.available_ram
    config["num_gpus"] = args.num_gpus
    
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def get_txt_files(directory) -> List[Path]:
    SUPPORTED_FORMATS = ['.txt']
    
    logs_files: List[Path] = []
    
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            logs_files.append(file)
            
    output: List[Path] = sorted(logs_files)

    return output

def read_and_preprocess_files(txt_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    def preprocess_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines -> double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        text = re.sub(r'\n ', '\n', text)  # Remove spaces at start of lines
        
        # Remove common file artifacts
        text = re.sub(r'\x0c', '', text)  # Form feed characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control chars
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")  # Smart quote
        text = text.replace('â€œ', '"')  # Smart quote
        text = text.replace('â€', '"')   # Smart quote
        text = text.replace('â€"', '—')  # Em dash
        text = text.replace('â€"', '–')  # En dash
        
        # Normalize quotes and dashes
        # text = re.sub(r'["""]', '"', text)  # Various quotes to standard
        # text = re.sub(r'['']', "'", text)  # Various apostrophes to standard
        # text = re.sub(r'[—–]', '-', text)   # Various dashes to hyphen
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots -> ellipsis
        text = re.sub(r'[!]{2,}', '!', text)   # Multiple exclamation -> single
        text = re.sub(r'[?]{2,}', '?', text)   # Multiple question -> single
        
        # Strip and ensure we don't have empty result
        text = text.strip()
        
        return text

    processed_files = {}
    failed_files = []

    for file_path in tqdm(txt_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_content = f.read()
            
            clean_content = preprocess_text(raw_content)
            
            if not clean_content:
                print(f"Skipping empty file: {file_path}")
                continue
            
            token_count = count_tokens(clean_content)
                        
            processed_files[file_path] = {
                "content": clean_content,
                "num_tokens": token_count,
                "file_path": str(file_path)
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Successfully processed: {len(processed_files)} files")

    return processed_files

def get_model_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = {
        "swap_space": math.floor((int(config["system_ram_gb"]))/4*3),
        "cpu_offload": 0,
        "tensor_parallel_size": int(config["num_gpus"])
    }

    if config["gpu_type"] == "H100":
        repo_id = "MaziyarPanahi/phi-4-GGUF"
        filename = "phi-4.fp16.gguf"
        model = hf_hub_download(repo_id, filename=filename)

        model_config.update({
            "model_name": model,
            "tokenizer" : "microsoft/phi-4",
            "max_model_len": 16000,
            "gpu_memory_utilization": 0.9,
            "quantization": "gguf",
        })
    elif config["gpu_type"] == "A5000":
        repo_id = "MaziyarPanahi/phi-4-GGUF"
        filename = "phi-4.Q4_K_M.gguf"
        model = hf_hub_download(repo_id, filename=filename)

        model_config.update({
            "model_name": model,
            "tokenizer" : "microsoft/phi-4",
            "max_model_len": 16000, 
            "gpu_memory_utilization": 0.9,
            "quantization": "gguf",
        })
    elif config["gpu_type"] == "RTX2080":
        repo_id = "MaziyarPanahi/phi-4-GGUF"
        filename = "phi-4.Q4_K_M.gguf"
        model = hf_hub_download(repo_id, filename=filename)

        model_config.update({
            "model_name": model,
            "tokenizer" : "microsoft/phi-4",
            "max_model_len": 8000, 
            "gpu_memory_utilization": 0.9,
            "quantization": "gguf",
        })
    
    return model_config

def process_chunking_decision(processed_files, model_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    def calculate_chunk_parameters(model_config: Dict[str, Any]) -> Dict[str, int]:
        max_model_len = model_config["max_model_len"]
        
        usable_context = int(max_model_len * 0.25 - 800)  # reserve 75% of context for output and prompt

        overlap_ratio = 0.1   # 10% overlap
        chunk_size = min(usable_context, 100000)
        
        overlap_size = int(chunk_size * overlap_ratio)
        
        return {
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "overlap_ratio": overlap_ratio,
            "usable_context": usable_context
        }
    
    def smart_text_splitter(text: str, chunk_size: int, overlap_size: int) -> List[str]:
        if not text:
            return []
        
        chunk_chars = chunk_size * 4
        overlap_chars = overlap_size * 4
        
        if len(text) <= chunk_chars:
            return [text]  # No chunking needed
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(text):
            end_pos = min(start_pos + chunk_chars, len(text))
            
            if end_pos < len(text):
                paragraph_break = text.rfind('\n\n', start_pos, end_pos)
                if paragraph_break > start_pos + chunk_chars * 0.7: 
                    end_pos = paragraph_break + 2
                else:
                    sentence_break = text.rfind('. ', start_pos, end_pos)
                    if sentence_break > start_pos + chunk_chars * 0.7:
                        end_pos = sentence_break + 2 
                    else:
                        line_break = text.rfind('\n', start_pos, end_pos)
                        if line_break > start_pos + chunk_chars * 0.7:
                            end_pos = line_break + 1
            
            chunk = text[start_pos:end_pos].strip()

            if chunk:
                chunks.append(chunk)
            
            if end_pos >= len(text):
                break  # We've reached the end
                
            # next_start = max(end_pos - overlap_chars, start_pos + chunk_chars // 2)
            next_start = max(end_pos - overlap_chars, start_pos + 1)

            if next_start <= start_pos:
                next_start = start_pos + max(1, chunk_chars // 4)

            start_pos = next_start
        
        return chunks

    chunk_params = calculate_chunk_parameters(model_config)

    files_needing_chunking = []
    files_no_chunking = []

    # Create a copy to avoid modifying the original during iteration
    result_files = processed_files.copy()
    
    for filename, file_data in result_files.items():
        if file_data["num_tokens"] > chunk_params["usable_context"]:
            files_needing_chunking.append(filename)
        else:
            files_no_chunking.append(filename)

    total_chunks_created = 0

    print(f"Processing {len(result_files)} files...")
    print(f"Files needing chunking: {len(files_needing_chunking)}")
    print(f"Files not needing chunking: {len(files_no_chunking)}")
    
    for filename in tqdm(result_files.keys(), desc="Chunking analysis"):
        file_data = result_files[filename]
        
        if file_data["num_tokens"] > chunk_params["usable_context"]:
            # print(f"Chunking file: {filename} ({file_data['num_tokens']} tokens)")

            chunks = smart_text_splitter(
                file_data["content"],
                chunk_params["chunk_size"],
                chunk_params["overlap_size"]
            )
            
            file_data["is_chunked"] = True
            file_data["chunks"] = chunks
            file_data["num_chunks"] = len(chunks)
            file_data["chunk_params"] = chunk_params.copy()
            
            chunk_tokens = []
            for i, chunk in enumerate(chunks):
                chunk_token_count = count_tokens(chunk)
                chunk_tokens.append(chunk_token_count)
            
            file_data["chunk_token_counts"] = chunk_tokens
            file_data["total_chunk_tokens"] = sum(chunk_tokens)
            
            total_chunks_created += len(chunks)
        else:
            file_data["is_chunked"] = False
            file_data["chunks"] = [file_data["content"]]  # Single "chunk" for consistency
            file_data["num_chunks"] = 1
            file_data["chunk_token_counts"] = [file_data["num_tokens"]]
            file_data["total_chunk_tokens"] = file_data["num_tokens"]
            file_data["chunk_params"] = chunk_params.copy()
    
    print(f"Total chunks created: {total_chunks_created}")
    print(f"Final number of files: {len(result_files)}")
    
    return result_files

def prepare_prompts(processed_txt_files: Dict[str, Dict[str, Any]],  prompt_prefix: str = "") -> Tuple[List[Tuple[str, str, int]], List[str]]:
    all_chunks: List[Tuple[str, str, int]] = []

    # print(f"Starting with {len(processed_txt_files)} files")

    files_with_chunks = 0
    files_without_chunks = 0
    total_chunks_found = 0

    for filename, data in processed_txt_files.items():
        chunks: List[str] = data.get("chunks", [])
        token_counts: List[int] = data.get("chunk_token_counts", [])


        # Debug missing data
        if not chunks:
            print(f"WARNING: No chunks found for file: {filename}")
            print(f"  Available keys: {list(data.keys())}")
            files_without_chunks += 1
            continue
            
        if not token_counts:
            print(f"WARNING: No token counts found for file: {filename}")
            print(f"  Available keys: {list(data.keys())}")
            files_without_chunks += 1
            continue

        # Check for length mismatch
        if len(chunks) != len(token_counts):
            print(f"WARNING: Length mismatch for file {filename}")
            print(f"  Chunks: {len(chunks)}, Token counts: {len(token_counts)}")
            # Take the minimum to avoid index errors
            min_length = min(len(chunks), len(token_counts))
            chunks = chunks[:min_length]
            token_counts = token_counts[:min_length]
            print(f"  Truncated both to length: {min_length}")
        
        file_chunk_count = 0
        # paired: List[Tuple[str, int]] = sorted(zip(chunks, token_counts), key=lambda x: x[1])
        paired: List[Tuple[str, int]] = list(zip(chunks, token_counts))

        for chunk, count in paired:
            # Skip empty chunks
            if not chunk.strip():
                print(f"WARNING: Empty chunk found in {filename}, skipping")
                continue
                
            # Skip zero token chunks
            if count <= 0:
                print(f"WARNING: Zero token chunk found in {filename}, skipping")
                continue

            all_chunks.append((filename, chunk, count))
            file_chunk_count += 1

        if file_chunk_count > 0:
            files_with_chunks += 1
            total_chunks_found += file_chunk_count
            # print(f"File {filename}: {file_chunk_count} chunks processed")
        else:
            files_without_chunks += 1
            print(f"WARNING: No valid chunks processed for file: {filename}")
    
    all_chunks.sort(key=lambda x: x[2])

    # prompts: List[str] = [prompt_prefix + chunk for _, chunk, _ in all_chunks]
    prompts: List[str] = [prompt_prefix + chunk for filename, chunk, count in all_chunks]

    print(f"Generated {len(prompts)} prompts for analyzing.")

    return all_chunks, prompts

def extract_data_and_generate_output_file(file_path):
    parts = file_path.parts
    newspaper = parts[2]
    year = parts[3]
    month = parts[4]
    day = Path(parts[5]).stem
    match = re.search(r'_(\d+)$', day)
    
    try:
        day = match.group(1)
    except Exception as e:
        print(f"Error {e} in {day} found in {file_path}")
        day = "15"

    date_string = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    date_string_suffix = f"{year}{month.zfill(2)}{day.zfill(2)}"
    filename = parts[5]

    output_path = str(file_path).replace("data/", "results/")
    output_path = output_path.replace("/txt/", "/csv/")
    output_path = output_path.replace(f"{newspaper}_{day.zfill(2)}", f"{newspaper}_{date_string_suffix}")
    output_path_correct = output_path.replace(".txt", ".csv")

    output_dir = Path(output_path_correct).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return newspaper, date_string, output_path_correct, output_path

def save_outputs_to_csv(outputs, all_chunks, config, processed_txt_files) -> int:
    def extract_code_block(text: str, language_hint: str = "json") -> str:
        pattern_lang = rf"```{language_hint}\n(.*?)```"
        match = re.search(pattern_lang, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str.strip())
                return parsed
            except json.JSONDecodeError as e:
                # print(f"JSON parsing error: {e}")
                return None
        else:
            print(f"No {language_hint} code block found")
            return None

    def clean_text(text: str) -> str:
        text = text.replace("\n", " ")   # replace line breaks with spaces
        text = text.replace('"', "")     # remove double quotes
        text = text.replace("'", "")     # remove single quotes
        return text

    no_outputs = 0

    for (filename, _, _), output_obj in zip(all_chunks, outputs):
        try:
            full_path = Path(processed_txt_files[filename]["file_path"])

            newspaper, date_string, output_path, output_path_incorrect = extract_data_and_generate_output_file(full_path)

            if hasattr(output_obj, "outputs") and output_obj.outputs and hasattr(output_obj.outputs[0], "text"):
                output = output_obj.outputs[0].text
            elif isinstance(output_obj, str):
                output = output_obj
            else:
                output = f"[Unexpected output type: {type(output_obj)}]"
            
            answer = extract_code_block(output)

            df = pd.DataFrame(answer, columns=["headline", "content"])
        
            if df.empty:
                raise ValueError(f"No valid articles extracted from JSON parsing.")
            
            df["headline"] = df["headline"].astype(str).map(clean_text)
            df["content"] = df["content"].astype(str).map(clean_text)

            df["newspaper"] = newspaper
            df["date"] = date_string

            if os.path.exists(output_path):
                df.to_csv(output_path, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
                print(f"Appended rows to existing CSV: {output_path}")
            else:
                no_outputs += 1
                df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
                print(f"Created new CSV: {output_path}")
        except Exception as e:
            with open(output_path_incorrect, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- ERROR ENTRY: {filename} ---\n\n")
                f.write(output)
            print(f"Error processing {filename}, appended raw output to {output_path_incorrect}. Error: {e}")
    
    return no_outputs

def initialize_model(model_config):
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        # seed=42,
        min_p=0.05,
        max_tokens=int(model_config["max_model_len"] * 0.75 - 800), # reserve 25% of context for input and prompt
        repetition_penalty=1.125
    )

    # sampling_params = SamplingParams(
    #     n=3,
    #     temperature=0.4,
    #     top_p=0.95,
    #     top_k=40,
    #     # seed=42,
    #     min_p=0.05,
    #     max_tokens=int(model_config["max_model_len"] * 0.75 - 800), # reserve 25% of context for input and prompt
    #     repetition_penalty=1.125,
    #     best_of=3
    # )

    llm = LLM(
        model=model_config["model_name"],
        tokenizer=model_config["tokenizer"],
        max_model_len=model_config["max_model_len"],
        gpu_memory_utilization=model_config["gpu_memory_utilization"],
        seed=42,
        dtype='float16',
        quantization=model_config["quantization"],
        swap_space=model_config["swap_space"],
        cpu_offload_gb=model_config["cpu_offload"], # TODO: Try more
        max_seq_len_to_capture=8192,
        tensor_parallel_size=model_config["tensor_parallel_size"],
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=True,
        enforce_eager=False
    )

    return llm, sampling_params

def generate_chat_messages(all_chunks):
    instructions = """
    ROLE: You are an expert newspaper content extractor and JSON formatter. You receive ONLY raw OCR text from a Spanish (Peruvian) newspaper page and must produce a clean, valid JSON output.

    INPUT: A single plain-text string (UTF-8) that is the OCR of one newspaper page.

    SCOPE & GOAL:
    Extract EVERY article with meaningful journalistic content and output them as a JSON array of objects with this exact schema:
    [
        {
            "headline": string (spanish-language) or "NA",
            "content": string (spanish-language) or "NA"
        }
    ]

    INCLUDE vs EXCLUDE:
    - Include news articles, opinion columns, interviews, features with substantive text.
    - Exclude page dates, section labels, weather, stock tables, TV grids, classifieds, obituaries lists, horoscopes, crosswords, pure photo captions, advertisements, subscription boxes, printer marks.

    DEFINITIONS & HEURISTICS:
    - "headline": Short, prominent line (often all caps). If multiple candidates, choose the one that best names the piece. Preserve original casing.
    - "content": The main article body. Merge broken lines into sentences/paragraphs. Exclude photo credits ("Foto:", "Crédito:"), image captions, graph labels and sidebar blurbs unless they clearly belong to the same article body.

    TEXT CLEANUP:
    - Preserve Spanish accents and punctuation; do NOT paraphrase.
    - Fix only clear OCR artifacts:
    - Join hyphenated line-break words (e.g., "demo-\\ncracia" → "democracia").
    - Remove page headers/footers and repeated section labels if detached from article text.
    - Collapse multiple spaces/newlines to single spaces inside fields.
    - Ensure values are valid JSON strings (escape double quotes properly).

    MULTI-ARTICLE SEGMENTATION:
    - Treat the page as possibly containing multiple articles.
    - Start a new article when a new headline-like line appears or when a byline follows a headline and then body text begins.
    - Maintain top-to-bottom reading order (as reflected in the OCR text order).

    MISSING DATA:
    - If a field is absent/unclear, use the literal string "NA".
    - Never fabricate names or subheadlines.

    OUTPUT FORMAT (STRICT):
    - Output ONLY valid JSON.
    - Output must be a top-level array of objects.
    - Each object must contain exactly two keys: "headline", "content".
    - Values must be JSON strings (use null nowhere — use "NA" instead).
    - No extra commentary, no preface, no Markdown formatting.

    VALIDATION BEFORE OUTPUT:
    - If no eligible articles, output an empty array [].
    - Ensure the JSON is syntactically valid and can be parsed without errors.

    EXAMPLE OUTPUT:
    ```json
    [
        {
            "headline": "El loco del martillo",
            "content": "Hoy en día, uno pensaría que..."
        },
        {
            "headline": "Contento por fin de cuarentena!",
            "content": "Estoy feliz porque..."
        },
        {
            "headline": "Urgente: Se busca perro perdido. Recompensa: 500 soles.",
            "content": "NA"
        },
        {
            "headline": "NA",
            "content": "El juez Huamani ordenó..."
        }
    ]
    ```

    WARNING:
    If you fail, the output will be unusable. Follow instructions EXACTLY.
    Always extract all articles. Focus on your role as a content extractor and JSON formatter.
    """

    user_prompt_template = "{ocr_text}"

    chat_messages = []
    for _, chunk, _ in all_chunks:
        chat_messages.append([
            {"role": "system", "content": str(instructions)},
            {"role": "user", "content": user_prompt_template.format(ocr_text=chunk)}
        ])

    return chat_messages

def main() -> None:
    args = parse_arguments()
        
    config = validate_arguments(args)
    model_config = get_model_configuration(config)

    txt_files = get_txt_files(config["input_folder"])

    # Skips files that have already been processed
    filtered_txt_files = []
    for txt_file in txt_files:
        _, _, output_path, _ = extract_data_and_generate_output_file(txt_file)
        csv_file = Path(output_path)
        if not csv_file.exists():
            filtered_txt_files.append(txt_file)
        else:
            print(f"Skipping {txt_file} - corresponding CSV file exists: {csv_file}")

    processed_txt_files = read_and_preprocess_files(filtered_txt_files)
    chunked_txt_files = process_chunking_decision(processed_txt_files, model_config)
    all_chunks, prompts = prepare_prompts(chunked_txt_files)

    check_gpu()

    time_start = time.time()
    llm, sampling_params = initialize_model(model_config)
    print(f"Model initialized in {time.time() - time_start:.2f} seconds")
    
    chat_messages = generate_chat_messages(all_chunks)

    time_start = time.time()
    check_gpu()
    outputs = llm.chat(chat_messages, sampling_params)
    no_outputs = save_outputs_to_csv(outputs, all_chunks, config, processed_txt_files)
    check_gpu()
    total_time = time.time() - time_start

    print(f"Generated {no_outputs} files in {total_time:.2f} seconds.")
    average_time_per_file = total_time / no_outputs if no_outputs > 0 else 1
    print(f"Average time per file (secs): {average_time_per_file:.2f} seconds")
    print(f"Average time per file (mins): {(average_time_per_file/60):.2f} minutes")
    average_time_year = (average_time_per_file/60)*365
    print(f"Estimated time per year (hours): {average_time_year:.2f} minutes (ca. {average_time_year/60:.2f}) hours")
    min_estimated_total = (average_time_year/60)*6*7
    print(f"Estimated total processing walk-time (min. hour): {min_estimated_total:.2f} hours. ")

    destroy_model_parallel()
    del llm 
    gc.collect()
    torch.cuda.empty_cache()

    check_gpu()

if __name__ == "__main__":
    main()