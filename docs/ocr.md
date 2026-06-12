# ocr.py

Extracts text from preprocessed images using multiple OCR engines.

---

## 📥 Input
- Images from `./results/images/preprocessed/`

## 📤 Output
- Extracted text stored in `./results/txt/extracted/ocr_results_log.txt`

---

## 🧠 Supported Engines
- Tesseract
- EasyOCR
- KerasOCR
- PaddleOCR
- docTR

---

## 🧪 Example
```bash
python ocr.py
python ocr.py --test
python ocr.py --help
```

---

## Notes
- Each image is processed by each OCR engine in sequence to prevent GPU memory issues
- Threaded Image Importation using `ThreadPoolExecutor`