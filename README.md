# cOCRomla: Framework for OCR and Computational Linguistic Analysis
A complete pipeline for transforming (newspaper) images into structured linguistic data with built-in evaluation.

## 🌟 Key Features
- **Multi-stage processing**: Image → Text → Structured Data → Analysis
- **Modular design**: Swap OCR engines, LLMs, or analyzers
- **Computational linguistics**: Built-in topic modeling & sentiment analysis
- **Evaluation suite**: Precision, recall, and F1 metrics

## 🔄 Pipeline Overview

| Stage | Description | Output |
|-------|-------------|--------|
| 🖼️ Preprocessing | Image enhancement (binarization, deskewing, noise removal) | Processed `.tiff` images |
| 🔤 OCR Extraction | Multi-engine text recognition (Tesseract, EasyOCR, PaddleOCR) | Raw `.txt` files with OCR results |
| 🧠 LLM Structuring | Text normalization & structuring using Ollama LLMs (phi4, llama4, etc.) | Structured `.csv`/`.json` per image (article) |
| 📊 Quality Evaluation | Comparison against gold standard using fuzzy matching (accuracy, F1, precision, recall) | `evaluation.csv` with metrics |
| 🌐 Semantic Analysis | Topic modeling via ensemble of Spanish BERT embeddings | `topics.csv` with topic clusters |
| 😊 Sentiment Analysis | Majority voting across Spanish sentiment models (SaBERT, Robertuito, UMUTeam) | `sentiment.csv` with labels/scores |

## 🛠️ Installation

```bash
# 1) Clone repository
# 2) Install CUDA, Ollama and tesseract
./setup/install_cuda_toolkit+drivers.sh
./setup/install_ollama.sh
./setup/install_tesseract.sh
# 3) # Install all dependencies for the complete pipeline:
cat requirements/*_requirements.txt | sort -u > requirements/all.txt
pip install -r requirements/all.txt
# If not, check ./requirements/{}_requirements.txt
# 4) Verify installations
# 5) Read the → Script-specific documentation: ./docs/{}.md
```

## 🖼️ Preprocessing → 📊 Quality Evaluation: Images to structured text

### Folder structure

```
/data/
├── images/ # Raw input (newspapers) images (.png, .tiff)
├── csv/ # Gold standards annotations for evaluation (.csv)
/results/
├── images/preprocessed/ # Enhanced images (.tiff)
├── txt/extracted/ # Raw OCR outputs (.txt)
├── csv/extracted/ # Structured output (articles) text (.csv)
├── csv/evaluation.csv # Quality metrics (F1 and Accuracy)
```

### Quick Start

```bash
# Run complete workflow:
chmod +x ./run_ocr_pipeline.sh
./run_ocr_pipeline.sh.sh 

# Or run stages manually:
python3 src/preprocess/preprocess.py --help
python3 src/ocr/ocr.py --help
python3 src/postprocess/postprocess.py --help
python3 src/evaluate/evaluate.py --help
```

### Evaluation Metrics (based on CGEC13-22)

Each article’s `headline`, `subheadline`, and `content` fields are compared against gold labels using fuzzy string similarity (via `difflib`). Predictions are converted to binary matches and evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Results are saved to: `./results/csv/evaluation.csv`

### Requirements

- Python 3.10+ and dependencies
- Tesseract (installed locally)
- Ollama (installed locally and initialized)

## 🌐 Semantic Analysis: Structured text to topics

### Sample Output

```
document_id,text,agreed_topic,agreed_topic_label,confidence
1,"El Barcelona ganó...",12,"Deportes, fútbol, liga",0.92
2,"El presidente anunció...",34,"Política, gobierno",0.87
```

## 😊 Sentiment Analysis: Structured text to label and score

### Sample Output

```
article_id,text,agreed_label,agreed_score,sabert_label,robertuito_score
1,"El mercado sube...","Positive",0.91,POS,0.94
2,"Crisis económica...","Negative",0.87,NEG,0.82
```

---

## 🧾 License
GPL License