"""
utils_preprocessing.py

Utility functions for image preprocessing tasks including reading, saving,
displaying images, and file discovery.

These utilities support preprocessing steps for an OCR pipeline that converts
newspaper page images into structured CSV data.

Author: @nicolasleonri (GitHub)
License: GPL
"""
import matplotlib.pyplot as plt
from typing import List, Union
from numpy import ndarray
from pathlib import Path
import numpy as np
import cv2


def read_image(path: Union[str, Path]) -> Union[ndarray, None]:
    """Reads an image from the given file path using OpenCV.

    Args:
        path (str or Path): Path to the image file.

    Returns:
        numpy.ndarray: Image as a NumPy array, or None if it fails.
    """
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def save_image(image: ndarray, path: Union[str, Path]) -> None:
    """Saves an image to the specified path.

    Args:
        image (numpy.ndarray): Image to save.
        path (str or Path): Destination path.
    """
    cv2.imwrite(str(path), image)


def show_image(image: ndarray) -> None:
    """Displays an image using matplotlib.

    Args:
        image (numpy.ndarray): Image to display.

    Returns:
        None
    """
    if image is None or image.size == 0:
        print("Image cannot be shown.")
        return None
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def get_image_files(directory: Union[str, Path]) -> List[Path]:
    """Gets a list of image files from a given directory.

    Args:
        directory (str or Path): Directory to search for image files.

    Returns:
        list: Sorted list of Path objects pointing to image files.
    """
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']
    image_files = []
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file)

    output = sorted(image_files)
    return output


def is_grayscale(image: ndarray) -> bool:
    """Checks whether the image is in grayscale format.

    Args:
        image (numpy.ndarray): Image to check.

    Returns:
        bool: True if grayscale, False otherwise.
    """
    return len(image.shape) == 2


def to_grayscale(image: ndarray) -> ndarray:
    """Converts an image to grayscale if it is not already.

    Args:
        image (numpy.ndarray): Color or grayscale image.

    Returns:
        numpy.ndarray: Grayscale image.
    """
    if not is_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
