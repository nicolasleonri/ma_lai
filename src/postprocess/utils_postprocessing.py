import re
import os
from collections import defaultdict
from typing import Dict, List, Any
import glob
import io
import base64


def image_to_base64png(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode("utf-8")
    return base64_string

def log_processing_info(log_file_path, processed_filename, config_number, ocr_name, model_display_name, time_elapsed):
    """Log processing information with immediate flush"""
    log_entry = f"File: {processed_filename} - Config: {config_number} - OCR: {ocr_name} - LLM: {model_display_name} - Time needed: {time_elapsed}s\n"
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)
        f.flush()  # Immediate flush to disk

def log_processing_info_olmo(log_file_path, processed_filename, config_number, model_display_name, time_elapsed, natural_text):
    """Log processing information with immediate flush"""
    log_entry = f"File: {processed_filename} - Config: {config_number} - OCR: {model_display_name} - Time needed: {time_elapsed}s - [{natural_text}]\n"
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)
        f.flush()  # Immediate flush to disk

def extract_filename_and_config(filepath):
    """Extract processed filename and config number in one line each"""
    filename = os.path.basename(filepath)
    config_number = int(re.search(r'_config(\d+)', filename).group(1)) if re.search(r'_config(\d+)', filename) else None
    processed_filename = str(re.sub(r'_config\d+', '', filename))
    # ocr_number = 
    return processed_filename, config_number

def parse_ocr_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match each entry
    pattern = r'File: (.*?) - Config: (.*?) - OCR: (.*?) - Time needed: (.*?) - \[(.*?)\]'
    # pattern = r'File: (.*?) - Time needed: (.*?) - Config: (.*?) - \[(.*?)\]'
    
    # Find all matches
    matches = re.findall(pattern, content, re.DOTALL)
    # print(len(matches))

    idx = 0
    results = defaultdict(dict)

    for file_path, config, ocr, time, text in matches:
        clean_text = ' '.join(text.split())
        results[idx] = {'filepath': str(file_path), 'config': str(config), 'ocr': str(ocr), 'text': str(clean_text)} # Config of OCR
        idx += 1
    
    return results

def parse_image_results(images_directory: str) -> Dict[int, Dict[str, Any]]:
    """Parse image files from a directory and create a results dictionary
    similar to parsed OCR format (without OCR content).

    Args:
        images_directory (str): Path to directory containing images.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary with image metadata.
    """
    image_results = defaultdict(dict)
    
    # Common image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_directory, ext)))
        image_files.extend(glob.glob(os.path.join(images_directory, ext.upper())))
    
    # Create results dictionary
    for idx, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        # Extract 'configX' pattern from filename, e.g. 'config0'
        match = re.search(r'config(\d+)', filename)
        config_value = match.group(1) if match else 'default'

        image_results[idx] = {
            'filepath': str(image_path),
            'config': config_value,
            'ocr': 'vlm',
            'text': ''  # Leave text empty for image-only entries
        }

    # print(f"Found {len(image_files)} images in {images_directory}")
    return image_results

def load_shared_inputs(ocr_log_path: str):
    ocr_results = parse_ocr_results(os.path.join(os.getcwd(), ocr_log_path))
    img_results = parse_image_results("./results/images/preprocessed")
    print(f"Loaded {len(ocr_results)} OCR results")
    return ocr_results, img_results

def extract_code_block(text: str, language_hint: str = "") -> str:
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