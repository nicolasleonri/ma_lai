# postprocess.py

Uses Ollama (local LLMs) to convert raw OCR text into structured article CSVs or JSON.

---

## 📥 Input
- OCR log: `./results/txt/extracted/ocr_results_log.txt`

## 📤 Output
- Structured `.csv` files: `./results/csv/extracted/`
- (Optional) structured `.json` files: `./results/txt/extracted/`

---

## 🧠 Models Used (via Ollama)
- phi4
- llama4
- gemma3
- qwen3
- deepseek-r1
- magistral

---

## ⚙️ Features
- Multithreaded CSV processing
- CSV or JSON output with metadata extraction
- `--test` flag to run mock/test LLM output
- `--help` flag for usage instructions

---

## 🧪 Example
```bash
python postprocess.py
python postprocess.py --test
python postprocess.py --help
```

---

## ⚠️ Notes
- Requires Ollama server running locally
