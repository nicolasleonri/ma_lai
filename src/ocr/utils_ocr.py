"""
utils_ocr.py

Utility functions for discovering image files for OCR processing.
"""
from pathlib import Path
from typing import List
import time
import torch
import gc


def get_image_files(directory: str) -> List[Path]:
    """Returns a list of supported image files from the given directory.

    Args:
        directory (str): Path to the folder containing image files.

    Returns:
        List[Path]: A list of valid image file paths.
    """
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']
    image_files = []
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file)
    return image_files


def wait_for_gpu_memory(min_memory_mb: int = 2048, max_wait_time: int = 300, check_interval: int = 5):
    """Wait until GPU has enough free memory.

    Args:
        min_memory_mb (int): Minimum required memory in MB
        max_wait_time (int): Maximum time to wait in seconds
        check_interval (int): How often to check in seconds
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory check")
        return True

    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            # Get GPU memory info
            torch.cuda.empty_cache()
            gc.collect()

            # Check memory for default GPU
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(
                device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)

            free_memory = total_memory - max(allocated_memory, cached_memory)
            free_memory_mb = free_memory / (1024 * 1024)

            print(f"GPU Memory - Total: {total_memory/(1024**3):.1f}GB, "
                  f"Free: {free_memory_mb:.0f}MB, Required: {min_memory_mb}MB")

            if free_memory_mb >= min_memory_mb:
                print("✓ Enough GPU memory available")
                return True
            else:
                print(
                    f"⏳ Waiting for GPU memory... ({free_memory_mb:.0f}MB < {min_memory_mb}MB)")
                time.sleep(check_interval)

        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            time.sleep(check_interval)

    print(f"⚠️ Timeout waiting for GPU memory after {max_wait_time}s")
    return False
