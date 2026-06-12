"""
preprocess.py

Preprocessing module for transforming image files into high-quality
inputs for OCR. It includes a full pipeline of binarization, skew correction,
and noise removal, each with multiple configurable techniques.

This script supports multiprocessing, configuration permutations, and
logging. A testing mode can be triggered with `--test`.

Author: @nicolasleonri (GitHub)
License: GPL
"""
from utils_preprocessing import read_image, save_image, show_image, get_image_files, to_grayscale
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional, Union
from skimage.filters import threshold_niblack
import matplotlib.pyplot as plt
import multiprocessing as mp
from numpy import ndarray
from pathlib import Path
import numpy as np
import itertools
import unittest
import math
import time
import cv2
import sys
import os
import gc


class Binarization:
    """Provides several binarization methods for thresholding grayscale images."""
    @staticmethod
    def none(image: ndarray) -> ndarray:
        gray = to_grayscale(image)
        return gray

    @staticmethod
    def basic(image: ndarray) -> ndarray:
        """Applies basic binary thresholding.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Binarized image.
        """
        gray = to_grayscale(image)
        _, output = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return output

    @staticmethod
    def otsu(image: ndarray, with_gaussian: bool = False) -> ndarray:
        """Applies Otsu's thresholding, optionally with Gaussian blur.

        Args:
            image (ndarray): Input image.
            with_gaussian (bool): Whether to apply Gaussian blur before thresholding.

        Returns:
            ndarray: Binarized image.
        """
        gray = to_grayscale(image)
        src = cv2.GaussianBlur(gray, (5, 5), 0) if with_gaussian else gray
        _, output = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return output

    @staticmethod
    def adaptive_mean(image: ndarray) -> ndarray:
        """Applies adaptive mean thresholding after median blur.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Binarized image.
        """
        gray = to_grayscale(image)
        medblur = cv2.medianBlur(gray, 5)
        return cv2.adaptiveThreshold(medblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def adaptive_gaussian(image: ndarray) -> ndarray:
        """Applies adaptive Gaussian thresholding after median blur.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Binarized image.
        """
        gray = to_grayscale(image)
        medblur = cv2.medianBlur(gray, 5)
        return cv2.adaptiveThreshold(medblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def yannihorne(image: ndarray, show: bool = False) -> ndarray:
        """Performs a custom binarization based on mean + std threshold and morphological cleaning.

        Args:
            image (ndarray): Input image.
            show (bool): Whether to normalize for visualization.

        Returns:
            ndarray: Binarized and cleaned image.
        """
        gray = to_grayscale(image)
        mean, std = cv2.meanStdDev(gray)
        threshold = mean[0][0] + std[0][0]
        _, output = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        output = output.astype(np.uint8)

        # Morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=2)
        if show == True:
            output = cv2.normalize(output, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return output
        else:
            return output

    @staticmethod
    def niblack(image: ndarray, show: bool = False, window_size: int = 25, k: float = -0.2) -> ndarray:
        """Applies Niblack thresholding and post-cleaning.

        Args:
            image (ndarray): Input image.
            show (bool): Whether to normalize for visualization.
            window_size (int): Size of the window used by Niblack.
            k (float): Constant that affects threshold calculation.

        Returns:
            ndarray: Binarized and cleaned image.
        """
        gray = to_grayscale(image)
        thresh_niblack = threshold_niblack(gray, window_size=window_size, k=k)
        binary = gray <= thresh_niblack

        output = binary.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=2)

        if show == True:
            output = cv2.normalize(output, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return output
        else:
            return output


class SkewCorrection:
    """Provides several techniques for correcting skew in document images."""
    @staticmethod
    def boxes(image: ndarray) -> ndarray:
        """Correct skew using bounding box angle from minimum area rectangle.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Rotated image with corrected skew.
        """
        gray = to_grayscale(image)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate around center with updated bounding box
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate the image to correct skew
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def hough_transform(image: ndarray) -> ndarray:
        """Correct skew by computing the average line orientation using Hough transform.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Rotated image.
        """
        def compute_skew(image: ndarray) -> float:
            gray = to_grayscale(image)
            edges = cv2.Canny(gray, 50, 200)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
            angle = 0.0

            if lines is None or len(lines) == 0:
                raise ValueError("No lines found in Hough transform.")

            try:
                nb_lines = len(lines)
            except TypeError as e:
                print(f"COMMON ERROR: {e}")

            for line in lines:
                angle += math.atan2(line[0][3]*1.0 - line[0]
                                    [1]*1.0, line[0][2]*1.0 - line[0][0]*1.0)

            angle /= nb_lines*1.0

            return angle * 180.0 / np.pi

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        try:
            M = cv2.getRotationMatrix2D(center, compute_skew(image), 1.0)
        except Exception as e:
            print(f"COMMON ERROR: {e} Returning original image...")
            return image

        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def moments(image: ndarray) -> ndarray:
        """Correct skew based on image moments.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Deskewed image.
        """
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        moments = cv2.moments(binary)

        skew_angle = 0.5 * \
            np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
        skew_angle = np.degrees(skew_angle)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def topline(image: ndarray) -> ndarray:
        """Correct skew using the top line of horizontal projection.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Rotated image.
        """
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        horizontal_projection = np.sum(binary, axis=1)
        topline_idx = np.argmax(horizontal_projection)

        angle = np.arctan2(topline_idx - binary.shape[0] // 2, binary.shape[1])
        angle = np.degrees(angle)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def scanline(image: ndarray) -> ndarray:
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(
                "COMMON ERROR: No contours found for scanline method. Returning original image...")
            return image

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int8(box)
        angle = rect[-1]
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


class NoiseRemoval:
    """Provides several noise removal methods for correcting binarized images."""
    @staticmethod
    def none(image: ndarray) -> ndarray:
        return image

    """Provides multiple filtering techniques to reduce noise in document images."""
    @staticmethod
    def mean_filter(image: ndarray, kernel_size: int = 3) -> ndarray:
        """Applies a mean (box) filter to the image.

        Args:
            image (ndarray): Input image.
            kernel_size (int): Size of the kernel.

        Returns:
            ndarray: Smoothed image.
        """
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def gaussian_filter(image: ndarray, kernel_size: int = 3, sigma: float = 0) -> ndarray:
        """Applies a Gaussian blur to the image.

        Args:
            image (ndarray): Input image.
            kernel_size (int): Size of the kernel (should be odd).
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            ndarray: Blurred image.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    @staticmethod
    def median_filter(image: ndarray, kernel_size: int = 3) -> ndarray:
        """Applies a median filter to the image.

        Args:
            image (ndarray): Input image.
            kernel_size (int): Size of the kernel (must be odd).

        Returns:
            ndarray: Filtered image.
        """
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def conservative_filter(image: ndarray, kernel_size: int = 3) -> ndarray:
        """Applies a conservative smoothing filter to preserve edges.

        Args:
            image (ndarray): Input grayscale image.
            kernel_size (int): Neighborhood size.

        Returns:
            ndarray: Smoothed image.
        """
        pad_size = kernel_size // 2
        padded = np.pad(image, pad_size, mode='edge')
        result = np.zeros_like(image)

        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                region = padded[i - pad_size:i + pad_size +
                                1, j - pad_size:j + pad_size + 1]
                min_val = np.min(region)
                max_val = np.max(region)
                if image[i - pad_size, j - pad_size] < min_val:
                    result[i - pad_size, j - pad_size] = min_val
                elif image[i - pad_size, j - pad_size] > max_val:
                    result[i - pad_size, j - pad_size] = max_val
                else:
                    result[i - pad_size, j - pad_size] = image[i -
                                                               pad_size, j - pad_size]
        return result

    @staticmethod
    def laplacian_filter(image: ndarray) -> ndarray:
        """Enhances edges using the Laplacian operator.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Edge-enhanced image.
        """
        laplacian = cv2.Laplacian(image, cv2.CV_8U)
        inverted_laplacian = 255 - laplacian # Invert to highlight dark edges on light background
        return inverted_laplacian

    @staticmethod
    def frequency_filter(image: ndarray) -> ndarray:
        """Applies a low-pass filter in the frequency domain.

        Args:
            image (ndarray): Grayscale image.

        Returns:
            ndarray: Filtered image.
        """
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2

        # Create circular mask
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    @staticmethod
    def crimmins_speckle_removal(image: ndarray) -> ndarray:
        """Reduces speckle noise using the Crimmins algorithm.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Denoised image.
        """
        output = image.copy().astype(np.int32)
        total_iterations = 2 * (image.shape[0] - 2) * (image.shape[1] - 2)

        current_iteration = 0
        for _ in range(2):
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    current_iteration += 1
                    current_pixel = output[i, j]
                    neighbors = [output[i-1, j], output[i+1, j],
                                 output[i, j-1], output[i, j+1]]
                    med = np.median(neighbors)
                    if abs(current_pixel - med) > abs(current_pixel - np.mean(neighbors)):
                        output[i, j] = med

        return output.astype(np.uint8)

    @staticmethod
    def unsharp_filter(image: ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> ndarray:
        """Applies an unsharp mask to enhance details.

        Args:
            image (ndarray): Input image.
            kernel_size (int): Gaussian blur kernel size.
            sigma (float): Gaussian standard deviation.
            amount (float): Sharpening intensity.
            threshold (int): Threshold to suppress noise enhancement.

        Returns:
            ndarray: Sharpened image.
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened


class ColumnExtraction:
    @staticmethod
    def crop_vertical_columns(binary_img: ndarray, min_col_width: int = 75) -> List[ndarray]:
        """
        Crop vertical columns from a preprocessed binary image using histogram analysis.
        Ensures complete coverage - no parts of the image are eliminated.

        Args:
            binary_img (ndarray): Binary image with text in black (0), background white (255).
            min_col_width (int): Minimum width in pixels to consider a column (filter noise).
            debug (bool): If True, shows intermediate images and plots.

        Returns:
            List[ndarray]: List of cropped column images as numpy arrays with complete coverage.
        """

        if np.mean(binary_img) < 128:
            binary_img = 255 - binary_img

        vertical_sum = np.sum(binary_img == 0, axis=0)

        threshold = np.max(vertical_sum) * 0.05  # tweak this if needed
        gaps = vertical_sum < threshold

        text_regions = []
        in_column = False
        start = 0
        for i, is_gap in enumerate(gaps):
            if not is_gap and not in_column:
                start = i
                in_column = True
            elif is_gap and in_column:
                end = i
                in_column = False
                if end - start >= min_col_width:
                    text_regions.append((start, end))

        crop_boundaries = []
        image_width = binary_img.shape[1]

        if not text_regions:
            return [binary_img]

        current_pos = 0

        for i, (text_start, text_end) in enumerate(text_regions):
            if i == 0:
                # First column: include everything from start to middle of gap after this column
                if i < len(text_regions) - 1:
                    next_text_start = text_regions[i + 1][0]
                    gap_middle = (text_end + next_text_start) // 2
                    crop_boundaries.append((0, gap_middle))
                    current_pos = gap_middle
                else:
                    # Only one column, take entire width
                    crop_boundaries.append((0, image_width))
                    current_pos = image_width
            elif i == len(text_regions) - 1:
                # Last column: from current position to end of image
                crop_boundaries.append((current_pos, image_width))
            else:
                # Middle columns: from current position to middle of next gap
                next_text_start = text_regions[i + 1][0]
                gap_middle = (text_end + next_text_start) // 2
                crop_boundaries.append((current_pos, gap_middle))
                current_pos = gap_middle

        # Crop columns ensuring complete coverage
        columns = []
        for start, end in crop_boundaries:
            col_img = binary_img[:, start:end]
            columns.append(col_img)

        return columns

    @staticmethod
    def crop_horizontal_columns(binary_img: np.ndarray, min_row_height: int = 60) -> List[np.ndarray]:
        """
        Crop horizontal sections from a preprocessed binary image using histogram analysis.
        Ensures complete coverage - no parts of the image are eliminated.

        Args:
            binary_img (ndarray): Binary image with text in black (0), background white (255).
            min_row_height (int): Minimum height in pixels to consider a row (filter noise).
            debug (bool): If True, shows intermediate images and plots.

        Returns:
            List[ndarray]: List of cropped row images as numpy arrays with complete coverage.
        """
        if np.mean(binary_img) < 128:
            binary_img = 255 - binary_img

        horizontal_sum = np.sum(binary_img == 0, axis=1)

        threshold = np.max(horizontal_sum) * 0.05
        gaps = horizontal_sum < threshold

        text_regions = []
        in_row = False
        start = 0
        for i, is_gap in enumerate(gaps):
            if not is_gap and not in_row:
                start = i
                in_row = True
            elif is_gap and in_row:
                end = i
                in_row = False
                if end - start >= min_row_height:
                    text_regions.append((start, end))

        # Handle case where row goes till end of image
        if in_row:
            end = len(gaps) - 1
            if end - start >= min_row_height:
                text_regions.append((start, end))

        crop_boundaries = []
        image_height = binary_img.shape[0]

        if not text_regions:
            return [binary_img]

        current_pos = 0

        for i, (text_start, text_end) in enumerate(text_regions):
            if i == 0:
                # First row: include everything from start to middle of gap after this row
                if i < len(text_regions) - 1:
                    next_text_start = text_regions[i + 1][0]
                    gap_middle = (text_end + next_text_start) // 2
                    crop_boundaries.append((0, gap_middle))
                    current_pos = gap_middle
                else:
                    # Only one row, take entire height
                    crop_boundaries.append((0, image_height))
                    current_pos = image_height
            elif i == len(text_regions) - 1:
                # Last row: from current position to end of image
                crop_boundaries.append((current_pos, image_height))
            else:
                # Middle rows: from current position to middle of next gap
                next_text_start = text_regions[i + 1][0]
                gap_middle = (text_end + next_text_start) // 2
                crop_boundaries.append((current_pos, gap_middle))
                current_pos = gap_middle

        # Crop rows ensuring complete coverage
        rows = []
        for start, end in crop_boundaries:
            row_img = binary_img[start:end, :]
            rows.append(row_img)

        return rows


def process_image_configuration(args: Tuple[Any, Path, Dict[str, Any], int, str, bool]) -> Dict[str, Any]:
    """
    Process a single image with a given preprocessing configuration.

    Args:
        args (tuple): Tuple of (image_data, image_file, config, idx, processed_dir).

    Returns:
        dict: Result dictionary with log and status.
    """
    image_data, image_file, config, idx, processed_dir, crop_columns = args

    try:
        processed_image = image_data.copy()  # Work on a copy
        techniques = []

        start_time = time.time()

        # Apply each preprocessing step
        for step, method in config["preprocess"]:
            processed_image = getattr(step, method)(processed_image)
            techniques.append(f"{step.__name__}.{method}")

        end_time = time.time()
        time_elapsed = end_time - start_time

        # Save the processed image
        filepath = os.path.join(
            processed_dir, f"{image_file.stem}_config{idx}.tiff")
        save_image(processed_image, filepath)

        # Create log entry
        new_filename = image_file.name.replace('.png', '.tiff')
        log_entry = f"File: {new_filename} - Config {idx}: {', '.join(techniques)} - Time needed: {time_elapsed}s\n"

        if crop_columns:
            new_folder = os.path.join(
                processed_dir, f"{image_file.stem}_config{idx}")
            os.makedirs(new_folder, exist_ok=True)

            vertical_columns = ColumnExtraction.crop_vertical_columns(
                processed_image)
            for i, vert_col in enumerate(vertical_columns):
                path_file_columns = os.path.join(
                    new_folder, f"{image_file.stem}_config{idx}_vert_#{i}.tiff")

                horizontal_columns = ColumnExtraction.crop_horizontal_columns(vert_col)
                for j, hor_col in enumerate(horizontal_columns):
                    path_file_columns = os.path.join(new_folder, f"{image_file.stem}_config{idx}_hor_#{i}_{j}.tiff")
                    cv2.imwrite(path_file_columns, hor_col)

        del processed_image
        del image_data
        gc.collect()

        return {
            'success': True,
            'log_entry': log_entry,
            'config_idx': idx,
            'techniques': techniques,
            'time_elapsed': time_elapsed,
            'image_name': image_file.name
        }

    except Exception as e:
        error_msg = f"ERROR processing {image_file.name} with config {idx}: {str(e)}\n"
        return {
            'success': False,
            'log_entry': error_msg,
            'config_idx': idx,
            'error': str(e),
            'image_name': image_file.name
        }


class TestPreprocessingPipeline(unittest.TestCase):
    """Unit tests for individual preprocessing methods."""

    def setUp(self):
        self.blank = np.full((100, 100), 255, dtype=np.uint8)  # white image
        self.noisy = self.blank.copy()
        np.random.seed(0)
        self.noisy[np.random.randint(0, 100, 200),
                   np.random.randint(0, 100, 200)] = 0

    def test_to_grayscale(self):
        color_img = cv2.cvtColor(self.blank, cv2.COLOR_GRAY2BGR)
        gray = to_grayscale(color_img)
        self.assertEqual(len(gray.shape), 2)

    def test_binarization_basic(self):
        bin_img = Binarization.basic(self.noisy)
        self.assertEqual(bin_img.dtype, np.uint8)
        self.assertTrue((bin_img == 0).any() or (bin_img == 255).any())

    def test_noise_removal_median(self):
        denoised = NoiseRemoval.median_filter(self.noisy, kernel_size=3)
        self.assertEqual(denoised.shape, self.noisy.shape)

    def test_skew_correction_boxes(self):
        corrected = SkewCorrection.boxes(self.blank)
        self.assertEqual(corrected.dtype, self.blank.dtype)


def process_single_image_all_configs(image_file: Path, configurations: List[Dict[str, Any]], processed_dir: str, crop_columns: bool, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run all preprocessing configurations on a single image in parallel.

    Args:
        image_file (Path): Path to the input image.
        configurations (List[Dict]): List of preprocessing configurations.
        processed_dir (str): Directory to save processed images.
        max_workers (int): Number of parallel workers.

    Returns:
        List[Dict]: List of result dicts.
    """
    print(f"Processing: {image_file.name}")

    # Read image once
    try:
        image = read_image(image_file)
        if image is None:
            raise ValueError("Image failed to load.")
    except Exception as e:
        print(f"Error reading {image_file.name}: {e}")
        return []

    # Prepare arguments for all configurations for this image
    args_list = [
        (image, image_file, config, idx, processed_dir, crop_columns)
        for idx, config in enumerate(configurations)
    ]

    results = []
    completed_count = 0
    total_configs = len(configurations)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(process_image_configuration, args): args
            for args in args_list
        }

        # Process completed tasks
        for future in as_completed(future_to_args):
            result = future.result()
            results.append(result)
            completed_count += 1

            # Progress update
            if completed_count % 5 == 0 or completed_count == total_configs:
                print(
                    f"Progress: {completed_count}/{total_configs} configurations completed")

    print(f"Completed processing {image_file.name}")
    print("-" * 50)
    return results


def print_help() -> None:
    """Displays help message for command-line usage."""
    help_text = """
    Preprocessing Pipeline
    --------------------------

    Run full preprocessing over document images using configurable:
        - Binarization
        - Skew correction
        - Noise removal

    CLI Usage:
        python preprocess.py                   # Default mode
        python preprocess.py --per-image       # Run each image separately
        python preprocess.py --global          # Global parallelism
        python preprocess.py --threads 8       # Custom thread count
        python preprocess.py --test            # Run unit tests
        python preprocess.py --help / -h       # Show help

    Input/Output:
        - Input images: ./data/images/
        - Results saved to: ./results/images/preprocessed/
        - Log file: ./logs/preprocess.out
    """
    print(help_text)


def main() -> None:
    """Main function to run preprocessing pipeline based on CLI arguments."""
    gc.enable()

    # CLI parsing
    max_threads = mp.cpu_count()
    processing_mode = "per-image"
    run_tests = False
    crop_columns = False

    # TODO: Add flag "--crop_columns" and run pipeline

    if len(sys.argv) > 1:
        if any(arg in ['--help', '--h', '-h'] for arg in sys.argv):
            print_help()
            return

        if "--test" in sys.argv:
            run_tests = True

        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == '--threads' and i + 1 < len(sys.argv):
                try:
                    max_threads = int(sys.argv[i + 1])
                except ValueError:
                    print("Invalid thread count. Using default.")
            elif arg == '--per-image':
                processing_mode = "per-image"
            elif arg == '--global':
                processing_mode = "global"
            elif arg == '--crop_columns':
                crop_columns = True

    if run_tests:
        print("Running unit tests...")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        return

    print(f"Using {max_threads} threads in {processing_mode} mode")

    # Define preprocessing methods
    # binarization_methods = ["none", "basic", "otsu", "adaptive_mean", "adaptive_gaussian", ] # "yannihorne", "niblack"
    # noise_removal_methods = ["none", "mean_filter", "gaussian_filter", "median_filter", "conservative_filter", "crimmins_speckle_removal", "laplacian_filter", "frequency_filter", "unsharp_filter"]
    binarization_methods = ["none", "basic", "otsu", "adaptive_mean"]
    noise_removal_methods = ["none", "mean_filter", "conservative_filter", "median_filter", "conservative_filter", "unsharp_filter"]

    # Generate all possible configurations
    configurations = [
        {
            "preprocess": [
                (Binarization, bin_method),
                # (SkewCorrection, skew_method),
                (NoiseRemoval, noise_method)
            ]
        }
        for bin_method, noise_method in itertools.product(
            binarization_methods, noise_removal_methods
        )
    ]

    print(f"Total configurations to test: {len(configurations)}")

    # Setup directories
    image_files = get_image_files("./data/images/")
    processed_dir = "./results/images/preprocessed/"
    log_file_path = os.path.join("./logs/", "preprocess.out")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Filter out PDF files (as in original)
    image_files = [f for f in image_files if "pdf" not in str(f)]
    print(f"Processing {len(image_files)} images")

    start_time = time.time()
    all_results = []

    if processing_mode == "per-image":
        # Process each image sequentially, but all configs for each image in parallel
        for image_file in image_files:
            results = process_single_image_all_configs(
                image_file, configurations, processed_dir, crop_columns, max_threads
            )
            all_results.extend(results)

    elif processing_mode == "global":
        # Process all image/configuration combinations globally in parallel
        print("Processing all combinations globally in parallel...")

        # Prepare all tasks
        all_tasks = []
        for image_file in image_files:
            try:
                image = read_image(image_file)
                for idx, config in enumerate(configurations):
                    all_tasks.append(
                        (image, image_file, config, idx, processed_dir, crop_columns))
            except Exception as e:
                print(f"Error reading {image_file.name}: {e}")

        print(f"Total tasks: {len(all_tasks)}")

        # Process all tasks
        with ProcessPoolExecutor(max_workers=max_threads) as executor:
            future_to_task = {
                executor.submit(process_image_configuration, task): task
                for task in all_tasks
            }

            completed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                all_results.append(result)
                completed += 1

                if completed % 5 == 0:
                    print(
                        f"Progress: {completed}/{len(all_tasks)} tasks completed")

    # Write all results to log file
    with open(log_file_path, 'w') as log_file:
        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]

        # Write successful results
        for result in successful_results:
            log_file.write(result['log_entry'])

        # Write failed results
        if failed_results:
            log_file.write("\n--- ERRORS ---\n")
            for result in failed_results:
                log_file.write(result['log_entry'])

    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    successful_count = len([r for r in all_results if r['success']])
    failed_count = len([r for r in all_results if not r['success']])

    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful processes: {successful_count}")
    print(f"Failed processes: {failed_count}")
    print(
        f"Average time per successful process: {total_time/max(successful_count, 1):.3f} seconds")
    print(f"Processing log saved to: {log_file_path}")
    print("="*60)


if __name__ == "__main__":
    main()
