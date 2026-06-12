# preprocess.py

Applies image enhancement techniques to prepare image files for OCR.

---

## 📥 Input
- Raw images (e.g. `.png`, `.tiff`) in `./data/images/`

## 📤 Output
- Preprocessed images saved to `./results/images/preprocessed/`
- Log file: `./logs/preprocess.out`

---

## ⚙️ Key Features
- Supports multiple preprocessing steps and configurations:
  - Binarization methods (Otsu, Niblack, adaptive, etc.)
  - Skew correction methods (Moments, Hough transform, Topline detection, Scanline)
  - Noise reduction filters (median, conservative smoothing, unsharp mask, etc.)
- Parallel processing (per-image or global mode)
- Command-line flags:
  -  `--threads` to control parallelism
  -  `--per-image` to process images individually
  -  `--test` to run unit tests for preprocessing modules
  -  `--help` to display usage instructions

---

## 🧪 Example
```bash
python preprocess.py --threads 4 --per-image
python preprocess.py --test
```