"""
ocr.py

This module runs OCR extraction using multiple backends in parallel, including:
- Tesseract
- Keras-OCR
- EasyOCR
- PaddleOCR
- docTR

The pipeline supports multithreaded execution and performance logging.

Author: @nicolasleonri (GitHub)
License: GPL
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Union
# from doctr.models import ocr_predictor
from utils_ocr import get_image_files, wait_for_gpu_memory
# from doctr.io import DocumentFile
from tempfile import NamedTemporaryFile
# from paddleocr import PaddleOCR
import multiprocessing as mp
from typing import Union
from queue import Queue
from PIL import Image
import numpy as np
import pytesseract
import threading
# import keras_ocr
import unittest
# import easyocr
import torch
import time
import sys
import re
import gc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TesseractOCR:
    """Wrapper for Tesseract OCR using pytesseract."""

    @staticmethod
    def process(image: Union[str, np.ndarray]) -> str:
        """Extracts text using Tesseract.

        Args:
            image (str or ndarray): Path to image or image array.

        Returns:
            str: Extracted text.
        """
        gc.collect()
        torch.cuda.empty_cache()
        wait_for_gpu_memory()

        result = pytesseract.image_to_string(
            image, lang='spa', config="--psm 3 --oem 1")
        result = result.replace("\n", " ")  # TODO: try without combining
        return result


# class KerasOCR:
#     """Wrapper for Keras-OCR pipeline."""

#     @staticmethod
#     def process(image: Union[str, np.ndarray]) -> str:
#         """Extracts text using Keras-OCR.

#         Args:
#             image (str or ndarray): Path to image.

#         Returns:
#             str: Extracted text.
#         """

#         gc.collect()
#         torch.cuda.empty_cache()

#         wait_for_gpu_memory()

#         detector = keras_ocr.detection.Detector(weights='clovaai_general')
#         recognizer = keras_ocr.recognition.Recognizer(
#             # alphabet=keras_ocr.recognition.DEFAULT_ALPHABET + 'ñáéíóúüÑÁÉÍÓÚÜ¿¡',
#             weights='kurapan'  # This recognizer performs better on Latin characters
#         )

#         pipeline = keras_ocr.pipeline.Pipeline(
#             detector=detector, recognizer=recognizer)
#         read_image = keras_ocr.tools.read(image)
#         prediction_groups = pipeline.recognize([read_image])
#         # prediction_groups is a list of (word, box) tuples
#         output = [str(y[0]) for i in prediction_groups for y in i]
#         return " ".join(output)

#     # TODO: Find model for spanish-spoken newspaper

#     # def finetune_detector(data_dir):
#     #     # https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_detector.html
#     #     # https://www.kaggle.com/discussions/general/243859
#     #     return None

#     # def finetune_recognizer(data_dir):
#     #     # https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_recognizer.html
#     #     return None


# class EasyOCR:
#     """Wrapper for EasyOCR pipeline."""

#     @staticmethod
#     def process(image: Union[str, np.ndarray]) -> str:
#         """Extracts text using EasyOCR.

#         Args:
#             image (str or ndarray): Path to image.

#         Returns:
#             str: Extracted text.
#         """
#         gc.collect()
#         torch.cuda.empty_cache()
#         wait_for_gpu_memory()

#         reader = easyocr.Reader(['es'], gpu=True)
#         result = reader.readtext(image)
#         output = [j for _, j, _ in result]
#         return " ".join(output)


# class PaddlePaddle:
#     """Wrapper for PaddleOCR."""

#     @staticmethod
#     def process(image: Union[str, np.ndarray]) -> str:
#         """Extracts text using PaddleOCR.

#         Args:
#             image (str or ndarray): Path to image.

#         Returns:
#             str: Extracted text.
#         """
#         gc.collect()
#         torch.cuda.empty_cache()
#         wait_for_gpu_memory()

#         ocr = PaddleOCR(
#             text_detection_model_name="PP-OCRv5_server_det",
#             text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
#             use_doc_orientation_classify=False, 
#             use_doc_unwarping=False, 
#             use_textline_orientation=True, 
#             lang='es')
        
#         # Handle TIFF conversion if image is a path
#         temp_path = None
#         if isinstance(image, str) and image.lower().endswith(".tiff"):
#             with Image.open(image) as img:
#                 img = img.convert("RGB")
#                 with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#                     img.save(tmp.name)
#                     temp_path = tmp.name
#                     image = temp_path
#                     print(temp_path)


#         result = ocr.predict(image)
#         output = [j[1][0] for i in result for j in i]

#         # Clean up temp image
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)

#         return " ".join(output)


# class docTR:
#     """Wrapper for the docTR OCR system."""

#     @staticmethod
#     def clean_spanish_text(text: str) -> str:
#         """Clean and enhance Spanish text recognition results.

#         Args:
#             text (str): Raw OCR text output.

#         Returns:
#             str: Cleaned Spanish text.
#         """
#         if not text:
#             return ""

#         # Common OCR corrections for Spanish
#         corrections = {
#             # Fix common character recognition errors
#             r'rn': 'm',  # Common OCR error
#             r'cl': 'd',  # Another common error
#             r'ii': 'ü',  # Umlaut recognition
#             r'n\s*~': 'ñ',  # Fix ñ recognition
#             r'N\s*~': 'Ñ',
#             r'n\s*-': 'ñ',
#             r'N\s*-': 'Ñ',

#             # Fix inverted punctuation marks
#             r'^\s*\?': '¿',  # Question mark at start
#             r'^\s*!': '¡',   # Exclamation at start
#             r'\?\s*$': '?',  # Question mark at end
#             r'!\s*$': '!',   # Exclamation at end

#             # Fix common Spanish words that are often misrecognized
#             r'\bque\b': 'que',
#             r'\bdel\b': 'del',
#             r'\bcon\b': 'con',
#             r'\bpor\b': 'por',
#             r'\bpara\b': 'para',
#             r'\besta\b': 'está',
#             r'\beson\b': 'son',
#             r'\btiene\b': 'tiene',
#             r'\bmás\b': 'más',

#             # Fix spacing issues
#             r'\s+': ' ',  # Multiple spaces to single space
#             r'\n\s*\n': '\n\n',  # Clean up multiple newlines
#         }

#         cleaned_text = text
#         for pattern, replacement in corrections.items():
#             cleaned_text = re.sub(pattern, replacement,
#                                   cleaned_text, flags=re.IGNORECASE)

#         # Fix accented characters that might be misrecognized
#         accent_fixes = {
#             'a\'': 'á', 'e\'': 'é', 'i\'': 'í', 'o\'': 'ó', 'u\'': 'ú',
#             'A\'': 'Á', 'E\'': 'É', 'I\'': 'Í', 'O\'': 'Ó', 'U\'': 'Ú',
#             'u"': 'ü', 'U"': 'Ü'
#         }

#         for error, correction in accent_fixes.items():
#             cleaned_text = cleaned_text.replace(error, correction)

#         return cleaned_text.strip()

#     @staticmethod
#     def extract_text_from_document(document: DocumentFile, preserve_structure: bool = True) -> str:
#         """Flattens a DocumentFile into a plain text string with Spanish text handling.

#         Args:
#             document (DocumentFile): Parsed document object.
#             preserve_structure (bool): Whether to preserve line breaks and structure.

#         Returns:
#             str: Combined text from all pages.
#         """
#         if preserve_structure:
#             # Preserve document structure with line breaks
#             text_parts = []
#             for page in document.pages:
#                 page_text = []
#                 for block in page.blocks:
#                     block_text = []
#                     for line in block.lines:
#                         line_text = " ".join(
#                             word.value for word in line.words if word.value.strip())
#                         if line_text.strip():
#                             block_text.append(line_text)
#                     if block_text:
#                         page_text.append("\n".join(block_text))
#                 if page_text:
#                     text_parts.append("\n\n".join(page_text))

#             return "\n\n".join(text_parts)
#         else:
#             # Simple flat text extraction
#             text = " ".join(
#                 word.value
#                 for page in document.pages
#                 for block in page.blocks
#                 for line in block.lines
#                 for word in line.words
#                 if word.value.strip()  # Filter out empty words
#             )
#             return text

#     @staticmethod
#     def process(image: Union[str, np.ndarray], preserve_structure: bool = True) -> str:
#         """Extracts text using docTR's predictor.

#         Args:
#             image (str or ndarray): Path to image.

#         Returns:
#             str: Extracted text.
#         """
#         gc.collect()
#         torch.cuda.empty_cache()
#         wait_for_gpu_memory()

#         model = ocr_predictor(
#             det_arch='db_resnet50',
#             reco_arch='crnn_vgg16_bn',
#             pretrained=True,
#             assume_straight_pages=True,  # Assume text is horizontal
#             straighten_pages=False,       # Straighten rotated text
#             detect_orientation=True,     # Detect text orientation
#             detect_language=False        # We handle language-specific processing ourselves
#         )
#         single_img_doc = DocumentFile.from_images(image)
#         result = model(single_img_doc)

#         output = docTR.extract_text_from_document(result, preserve_structure)

#         # Apply Spanish-specific text cleaning
#         cleaned_output = docTR.clean_spanish_text(output)

#         return cleaned_output


class TestOCRPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate a simple image with text: 'Test OCR line'"""
        from PIL import ImageDraw, ImageFont

        cls.test_text = "Test OCR line"
        cls.test_dir = "./results/images/"
        cls.test_file_path = os.path.join(cls.test_dir, "image_test.png")
        os.makedirs(cls.test_dir, exist_ok=True)

        image = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        except IOError:
            font = ImageFont.load_default()

        draw.text((10, 30), cls.test_text, fill='black', font=font)
        image.save(cls.test_file_path)

    @classmethod
    def tearDownClass(cls):
        """Remove test image"""
        os.remove(cls.test_file_path)

    def check_ocr_output(self, method, method_name):
        """Helper: run method and assert it returns expected text"""
        output = method(self.test_file_path)
        self.assertIsInstance(
            output, str, f"{method_name} returned non-string output")
        self.assertTrue(
            any(word.lower() in output.lower()
                for word in ["test", "ocr", "line"]),
            f"{method_name} failed to detect expected text"
        )

    def test_tesseract(self):
        from ocr import TesseractOCR
        self.check_ocr_output(TesseractOCR.process, "Tesseract")

    def test_keras_ocr(self):
        from ocr import KerasOCR
        self.check_ocr_output(KerasOCR.process, "KerasOCR")

    def test_easyocr(self):
        from ocr import EasyOCR
        self.check_ocr_output(EasyOCR.process, "EasyOCR")

    def test_paddleocr(self):
        from ocr import PaddlePaddle
        self.check_ocr_output(PaddlePaddle.process, "PaddleOCR")

    def test_doctr(self):
        from ocr import docTR
        self.check_ocr_output(docTR.process, "docTR")


def process_single_ocr(image_path: str, method_name: str, method: Callable[[Union[str, np.ndarray]], str], log_queue: Queue, is_cropped_folder: bool) -> Dict[str, Union[str, float]]:
    """Processes a single image with a given OCR method.

    Args:
        image_path (str): Path to the image.
        method_name (str): Name of the OCR method.
        method (callable): Method to process the image.
        log_queue (Queue): Thread-safe queue for logging.

    Returns:
        dict: Result dictionary containing text and metadata.
    """
    start = time.time()
    output = ""
    error_msg = ""

    try:
        if is_cropped_folder:
            folder_path = os.path.splitext(image_path)[0]

            if not os.path.isdir(folder_path):
                raise ValueError(f"Expected folder path, got: {folder_path}")
            
            # Get all image files and sort them
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            image_files.sort()  # Sort to maintain order
            
            combined_text = []
            for img_file in image_files:
                # print(f"Processing cropped image: {img_file}")
                img_path = os.path.join(folder_path, img_file)
                try:
                    text = method(img_path)
                    if text.strip():  # Only add non-empty text
                        combined_text.append(text.strip())
                except Exception as e:
                    log_queue.put(f"Error processing {img_file} in {os.path.basename(img_file)}: {str(e)}")
            
            output = " ".join(combined_text)  # Simple concatenation with spaces

            method_name = method_name + "_cropped"
        else:
            # Process single image (original behavior)
            output = method(image_path)
    except ValueError as ve:
        error_msg = f'COMMON ERROR: {ve}' if "unable to read" in str(
            ve) else f'NEW ERROR: {ve}'
    except Exception as e:
        if "CUDA out of memory" in str(e) or "32-bit samples" in str(e):
            error_msg = f'COMMON ERROR: {e}'
        else:
            error_msg = f'NEW ERROR: {e}'

    end = time.time()
    time_elapsed = end - start

    result = {
        'image_file': os.path.basename(image_path),
        'method_name': method_name,
        'time_elapsed': time_elapsed,
        'output': output,
        'error_msg': error_msg
    }

    log_queue.put(result)  # Thread-safe logging

    return result


def log_writer(log_queue: Queue, log_file_path: str, total_tasks: int) -> None:
    """Continuously writes OCR results to log file.

    Args:
        log_queue (Queue): Queue with results.
        log_file_path (str): Path to the log file.
        total_tasks (int): Total number of tasks expected.
    """
    completed_tasks = 0

    with open(log_file_path, 'w') as log_file:
        while completed_tasks < total_tasks:
            try:
                result = log_queue.get(timeout=1)

                processed_filename = str(re.sub(r'_config\d+', '', result['image_file']))
                config_number = int(re.search(r'_config(\d+)', result['image_file']).group(1)) if re.search(r'_config(\d+)', result['image_file']) else None

                log_file.write(f"File: {processed_filename} - "
                               f"Config: {config_number} - "
                               f"OCR: {result['method_name']} - "
                               f"Time needed: {result['time_elapsed']} - "
                               f"[{result['output']}]\n")
                log_file.flush()

                if result['error_msg']:
                    print(f"Error: {result['error_msg']}")

                completed_tasks += 1
                log_queue.task_done()

            except:
                continue  # No log yet, keep looping


def print_help() -> None:
    """Display usage instructions for the OCR script."""
    help_text = """
    OCR Pipeline Help
    -----------------
    This script runs OCR extraction using multiple engines in parallel.

    Supported OCR Engines:
        - TesseractOCR
        - KerasOCR
        - EasyOCR
        - PaddleOCR
        - docTR

    Usage:
        python ocr.py                 Run full OCR pipeline on images in ./results/images/preprocessed/
        python ocr.py --test          Run unit tests (generates a test image and validates OCR output)
        python ocr.py --help | -h     Show this help message

    Input:
        - Image files should be placed in: ./results/images/preprocessed/

    Output:
        - OCR text is logged to: ./results/txt/extracted/ocr_results_log.txt
    """
    print(help_text)


def main() -> None:
    
    """Main entry point for OCR pipeline."""
    max_threads = mp.cpu_count()
    use_cropped_folder = False

    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return

    if '--test' in sys.argv:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        return
    
    if '--cropped_folders' or '--cropped_folder' in sys.argv:
        use_cropped_folder = True
        print("Using cropped folder for OCR processing.")

    ocr_methods = {
        "TesseractOCR": TesseractOCR.process,
        # "KerasOCR": KerasOCR.process,
        # "EasyOCR": EasyOCR.process,
        # "PaddleOCR": PaddlePaddle.process,
        # "docTR": docTR.process
    }

    # Define the directory containing image files
    image_files = get_image_files("./results/images/preprocessed/")
    processed_dir = "./results/txt/extracted/"
    log_file_path = os.path.join(processed_dir, "ocr_results_log.txt")
    if use_cropped_folder == True:
        log_file_path = os.path.join(processed_dir, "ocr_cropped_results_log.txt")
    os.makedirs(processed_dir, exist_ok=True)

    # Create thread-safe queue for logging
    log_queue = Queue()

    # Calculate total number of tasks
    total_tasks = len(image_files) * len(ocr_methods)
    print(
        f"Starting processing of {len(image_files)} images with {len(ocr_methods)} methods")
    print(f"Total tasks: {total_tasks}")

    # Start log writer thread
    log_thread = threading.Thread(
        target=log_writer,
        args=(log_queue, log_file_path, total_tasks)
    )
    log_thread.start()

    print(f"Using {max_threads} threads")

    def process_by_image():
        """Process images sequentially to avoid GPU memory conflicts."""

        total_tasks = len(image_files) * len(ocr_methods)
        completed = 0

        print(
            f"Starting sequential processing of {len(image_files)} images with {len(ocr_methods)} methods")
        print(f"Total tasks: {total_tasks}")

        start_time = time.time()

        # Process each image with each OCR method sequentially
        for image_idx, image_file in enumerate(image_files):
            image_path = os.path.join(os.getcwd(), image_file)

            print(
                f"\n📄 Processing image {image_idx + 1}/{len(image_files)}: {image_file}")

            for method_idx, (method_name, method) in enumerate(ocr_methods.items()):
                try:
                    print(
                        f"  🔍 Running {method_name}... ({method_idx + 1}/{len(ocr_methods)})")

                    task_start = time.time()
                    result = process_single_ocr(
                        image_path, method_name, method, log_queue, is_cropped_folder=use_cropped_folder)
                    task_duration = time.time() - task_start

                    completed += 1

                    # Progress update
                    progress_percent = (completed / total_tasks) * 100
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed if completed > 0 else 0
                    estimated_remaining = (
                        total_tasks - completed) * avg_time_per_task

                    print(
                        f"    ✅ {method_name} completed in {task_duration:.1f}s")
                    print(
                        f"    📊 Progress: {completed}/{total_tasks} ({progress_percent:.1f}%)")

                    if completed % 5 == 0 or method_idx == len(ocr_methods) - 1:
                        print(
                            f"    ⏱️  Elapsed: {elapsed_time:.1f}s, Est. remaining: {estimated_remaining:.1f}s")

                except Exception as e:
                    completed += 1
                    print(f"    ❌ {method_name} failed: {e}")

                    # Log the error but continue processing
                    try:
                        log_queue.put(
                            f"ERROR - {method_name} on {image_file}: {str(e)}")
                    except:
                        pass

        total_time = time.time() - start_time
        print(f"\n🎉 All processing completed!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per task: {total_time/total_tasks:.1f}s")

    start_time = time.time()

    process_by_image()

    log_thread.join()

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processing log saved to: {log_file_path}")
    print("="*60)


if __name__ == "__main__":
    main()
