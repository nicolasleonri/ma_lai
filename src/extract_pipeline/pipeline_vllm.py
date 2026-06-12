from vllm import LLM, SamplingParams
from pathlib import Path
import subprocess
import argparse
import pandas as pd
import time
import csv
from io import StringIO
import random
import os
import gc
import re

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    min_p=0.0,
    max_tokens=8192,
    n=1,
    seed=42,
    repetition_penalty=1.25 # Slight repetition penalty
)

instructions="""
ROLE: You are an expert newspaper content extractor and CSV formatter. You receive ONLY raw OCR text from a Spanish (Peruvian) newspaper page and must produce a clean, valid CSV.

INPUT: A single plain-text string (UTF-8) that is the OCR of one newspaper page.
SCOPE & GOAL: Extract EVERY article with meaningful journalistic content and structure it into rows with these fields:
- headline (string or "NA")
- subheadline (string or "NA")
- author (string or "NA")
- content (string or "NA")

INCLUDE vs EXCLUDE:
- Include n ews articles, opinion columns, interviews, features with substantive text.
- Exclude tiny notices, page dates, section labels (e.g., Política, Economía), weather, stock tables, TV grids, classifieds, obituaries lists, horoscopes, crosswords, pure photo captions, advertisements, public-service announcements, subscription boxes, page numbers, URLs, social handles, printer marks.

DEFINITIONS & HEURISTICS:
- Headline: Short, prominent line (often all caps). If multiple candidates, choose the one that best names the piece. Preserve original casing.
- Subheadline (bajada/copete): Explanatory line under/after headline; shorter than content; not a caption.
- Author (byline): Lines starting with or containing markers such as: "Por", "POR", "Por:", "Redacción", "La Seño", "La Seño María", "Crónica", "Corresponsal", "Columna", "Agencia", "AFP", "EFE", "Reuters". If multiple authors, keep as they appear. Do NOT infer from quoted speakers inside the content.
- Content: The main article body. Merge broken lines into sentences/paragraphs. Exclude photo credits ("Foto:", "Crédito:"), image captions, graph labels, and sidebar blurbs unless they clearly belong to the same article body.

TEXT CLEANUP:
- Preserve Spanish accents and punctuation; do NOT paraphrase.
- Fix only clear OCR artifacts:
  - Join hyphenated line-break words (e.g., "demo-\ncracia" → "democracia").
  - Remove page headers/footers and repeated section labels if detached from article text.
  - Collapse multiple spaces/newlines to single spaces *inside* fields.
- Replace any semicolons inside fields with commas to protect the CSV delimiter.
- Escape any double quotes inside fields by doubling them (RFC 4180): `"` → `""`.

MULTI-ARTICLE SEGMENTATION
- Treat the page as possibly containing multiple articles.
- Start a new article when a new headline-like line appears or when a byline follows a headline and then body text begins.
- Maintain top-to-bottom reading order (as reflected in the OCR text order).
- If an item is too short (e.g., < 200 characters of body text) and reads like a brief, notice, or caption, exclude it.

MISSING DATA
- If a field is absent/unclear, write "NA".
- Never fabricate names or subheadlines.

OUTPUT FORMAT (STRICT)
- Output ONLY a valid CSV; no preface, no explanations.
- Delimiter: semicolon `;`
- Header row first and always:
  "headline";"subheadline";"author";"content"
- Each subsequent row = one article.
- Enclose every field in double quotes.
- Do not insert semicolons inside fields (replace them with commas).
- Escape internal quotes by doubling them.
- Newlines are not allowed inside fields; replace internal newlines with single spaces.

VALIDATION BEFORE OUTPUT
- At least the header must be present. If no eligible articles, output only the header.
- Ensure the number of fields per row is exactly 4.
- Ensure all rows are properly quoted and separated by newlines.

EXAMPLE:
"headline";"subheadline";"author";"content"
"El loco del martillo";"NA";"La Seño María";"Hoy en día, uno pensaría que..."
"Contento por fin de cuarentena";"Habla Trome";"Ismael Lazo, Vecino de San Luis";"Estoy feliz porque..."
"""

def extract_prompt_length_from_error(error_message):
    """Extract the actual prompt length from the error message"""
    match = re.search(r'decoder prompt \(length (\d+)\)', str(error_message))
    if match:
        return int(match.group(1))
    return None

def chunk_text_by_tokens(text, max_input_tokens):
    """Split text into chunks based on token limits"""
    # Rough estimation: 1 token ≈ 4 characters for most models
    chars_per_token = 4
    max_chars = max_input_tokens * chars_per_token
    
    if len(text) <= max_chars:
        return [text]
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_code_block(text: str, language_hint: str = "csv") -> str:
    """Extracts a code block (e.g., CSV) from a markdown-formatted LLM response.

    Args:
        text (str): Full response string from LLM.
        language_hint (str, optional): Language label to look for (e.g., "csv").

    Returns:
        str: Cleaned code block string (e.g., CSV content).
    """
    if language_hint:
        pattern_lang = rf"```{language_hint}\n(.*?)```"
        match = re.search(pattern_lang, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    pattern_any = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_any, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    print("GPU Memory:", result.stdout.strip())

def get_logs_files(directory, divided=False):
    if divided == False:
        SUPPORTED_FORMATS = ['.csv', '.tiff']
    else:
        SUPPORTED_FORMATS = ['.txt']
    
    logs_files = []
    
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            logs_files.append(file)
            
    output = sorted(logs_files)
    return output

def main():
    # Token budget calculation
    MAX_MODEL_TOKENS = 16384
    OUTPUT_TOKENS = 8192
    SYSTEM_PROMPT_TOKENS = 350
    MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - OUTPUT_TOKENS - SYSTEM_PROMPT_TOKENS 

    parser = argparse.ArgumentParser(description='Preprocessor for document images.')
    parser.add_argument('-f', '--input_folder', required=True, help='Folder with OCR results')
    parser.add_argument('-n', '--newspaper', required=True, help='Newspaper name (required)')
    parser.add_argument('-i', '--input_type', choices=['complete', 'divided'], required=True, help='Type of input (required)')

    args = parser.parse_args()

    if args.input_type == 'divided':
        print(f"Input folder: {str(args.input_folder)}")
        log_files = get_logs_files(str(args.input_folder), True)
        
        log_dir = './logs/already_processed'
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{args.newspaper}.log")

        if os.path.exists(log_path):
            processed_files = set()

            with open(log_path, 'r', encoding='utf-8') as f_log:
                for line in f_log:
                    processed_files.add(line.strip())

            remaining_files = [f for f in log_files if str(f) not in processed_files]

            sample_size = max(1, len(remaining_files)//4)
            selected_files = random.sample(remaining_files, k=sample_size)

            with open(log_path, 'a', encoding='utf-8') as log_f:
                for fpath in selected_files:
                    log_f.write(f"{fpath}\n")
        else:
            selected_files = random.sample(log_files, k=len(log_files)//4)

            log_path = os.path.join(log_dir, f"{args.newspaper}.log")
            
            with open(log_path, 'w', encoding='utf-8') as log_f:
                for fpath in selected_files:
                    log_f.write(f"{fpath}\n")

        txt_files = []
        contents = []

        for txt_file in selected_files:
            txt_files.append(str(txt_file))
            with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                contents.append(f.read())
                
        combined_df = pd.DataFrame({
        'filename': txt_files,
        'extracted_text': contents
        })
    else:
        log_files = get_logs_files(str(args.input_folder))
        dfs = []
        for csv_file in log_files:
            df = pd.read_csv(csv_file)
            df = df[df['extracted_text'].notna() & df['extracted_text'].str.strip().astype(bool)]
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Total files: {len(combined_df)}")

    check_gpu()

    llm = LLM(
        model="unsloth/phi-4-unsloth-bnb-4bit",
        tensor_parallel_size=1,
        max_num_seqs=8192,
        enable_prefix_caching=True,
        enforce_eager=False,
        swap_space=16,
        max_num_batched_tokens=16384,
        max_model_len=16384,
        disable_log_stats=True,
        gpu_memory_utilization=0.875,
        cpu_offload_gb = 20,
        block_size=256, 
        quantization="bitsandbytes",
        enable_chunked_prefill=True
    )

    check_gpu()

    time_start = time.time()

    for idx, val in enumerate(combined_df["extracted_text"].tolist()):
        try:
            conversation = [
                {"role": "system", "content": str(instructions)},
                {"role": "user", "content": str(val)},
            ]
            filename = combined_df["filename"].tolist()[int(idx)]
            path = Path(str(filename))
            
            parts = path.parts
            if args.input_type == 'divided':
                idx = parts.index("txt")
                section = parts[idx + 1]
                year, month = parts[idx + 2: idx + 4]
                basename = path.name 
                day = basename.split("_")[1].split(".")[0]
                date_str = f"{year}/{month}/{day}"
            else:
                idx = parts.index("preprocessed")
                section = parts[idx + 1]
                year, month, day = parts[idx + 2: idx + 5]
                date_str = f"{year}/{month}/{day}"

            if str(section) != str(args.newspaper):
                print("Section found is not newspaper given.")
                continue

            parts = list(path.parts)
            if args.input_type == 'divided':
                idx = parts.index("txt")
                parts[idx] = "csv"
                idx = parts.index("data")
                parts[idx] = "results"
                csv_filename = f"{args.newspaper}_{year}_{month}_{day}.csv"
                output_file = Path(*parts).with_suffix(".csv")
                output_file = output_file.parent / csv_filename
            else:
                idx = parts.index("images")
                parts[idx] = "csv"
                idx = parts.index("preprocessed")
                parts[idx] = "postprocessed"
                output_file = Path(*parts).with_suffix(".csv")
            
            os.makedirs(output_file.parent, exist_ok=True)

            if output_file.exists():
                print("Output already exists.")
                continue

            outputs = llm.chat(conversation, sampling_params)
            answer = outputs[0].outputs[0].text.strip()

            answer = extract_code_block(answer, language_hint="csv")
            check_gpu()

            f = StringIO(answer)
            reader = csv.reader(f, delimiter=';', quotechar='"')
            rows = list(reader)
            df = pd.DataFrame(rows[1:], columns=rows[0])  # assume first row is header
            df[["newspaper"]] = section
            df[["date"]] = date_str
            df.to_csv(str(output_file), index=False, quoting=csv.QUOTE_ALL)

            del df, outputs
            gc.collect()
            
            check_gpu()
            print(f"✓ Extracted text to {output_file.name}")
        except Exception as e:
            if "longer than the maximum model length" in str(e):
                actual_prompt_length = extract_prompt_length_from_error(str(e))
                print(f"Text too long ({actual_prompt_length} tokens), chunking: {filename}")
                safe_input_tokens = MAX_INPUT_TOKENS * 0.9
                
                chunks = chunk_text_by_tokens(str(val), int(safe_input_tokens))
                chunk_results = []
                
                for chunk_idx, chunk in enumerate(chunks):
                    print(f"Processing chunk: {int(chunk_idx)+1}/{len(chunks)}")

                    chunk_conversation = [
                        {"role": "system", "content": str(instructions)},
                        {"role": "user", "content": str(chunk)},
                    ]
                    
                    chunk_outputs = llm.chat(chunk_conversation, sampling_params)
                    chunk_result = chunk_outputs[0].outputs[0].text.strip()
                    chunk_result = extract_code_block(chunk_result, language_hint="csv")
                    check_gpu()
                    
                    try:
                        f = StringIO(chunk_result)
                        reader = csv.reader(f, delimiter=';', quotechar='"')
                        rows = list(reader)
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        df[["newspaper"]] = section
                        df[["date"]] = date_str

                        print(df)

                        if output_file.exists():
                            existing_df = pd.read_csv(str(output_file))
                            combined_df_to_csv = pd.concat([existing_df, df], ignore_index=True)
                            combined_df_to_csv.to_csv(str(output_file), index=False, quoting=csv.QUOTE_ALL)
                        else:
                            df.to_csv(str(output_file), index=False, quoting=csv.QUOTE_ALL)
    
                        print(f"✓ Extracted chunk {int(chunk_idx)+1}/{len(chunks)} from {output_file.name}")
                    except Exception as e:
                        print(f"Error getting csv from {output_file.name} in chunk {int(chunk_idx)+1}/{len(chunks)}")
                        continue
                
                del chunks, chunk_outputs
                gc.collect()
                check_gpu()
                print(f"✓ Extracted text to {output_file.name}")
            else:
                del df, outputs
                gc.collect()
                check_gpu()                 
                filename = combined_df["filename"].tolist()[int(idx)]
                print(f"✗ Failed to process {filename}: {e}")

    total_time = time.time() - time_start
    check_gpu()

    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Processed {len(combined_df)} prompts in {total_time:.2f} seconds")
    print(f"Avg per (text) input: {total_time / len(combined_df):.2f} sec")


if __name__ == "__main__":
    main()