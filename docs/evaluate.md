# evaluate.py

Compares model-generated article outputs against gold-standard annotations using structured similarity metrics to retrieve Accuracy and F1-Score metrics.

---

## 📥 Input

- **Gold Standard CSVs**: `./data/csv/*_goldstandard.csv`
- **Predicted Article CSVs**: `./results/csv/extracted/*.csv`

Each gold standard file should be named like: `newspaper#date#page_goldstandard.csv`

---

## 📤 Output

- **Evaluation Results**: `./results/csv/evaluation.csv`

The output CSV includes:
- Filename stem
- Config ID
- OCR engine used
- LLM model name
- Accuracy, Precision, Recall, F1-score

---

## ⚙️ How It Works

1. Headlines are matched using fuzzy string similarity.
2. Each predicted article is aligned with a gold article.
3. Similarity is computed for fields: headline, subheadline, content.
4. Binary labels are generated (`1 = match`, `0 = mismatch`) based on a similarity threshold (default = 0.8).
5. Scikit-learn metrics are computed across all fields and articles.

---

## 🧪 Commands

```bash
python evaluate.py            # Run evaluation on all predictions
python evaluate.py --test     # Run unit test on a synthetic example
python evaluate.py --help     # Show help message
```

---

## 📦 Dependencies

- scikit-learn
- numpy
- fuzzywuzzy
- difflib (standard lib)