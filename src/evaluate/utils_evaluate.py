"""
utils_evaluate.py

Evaluation utilities for comparing structured article data (CSV/JSON)
against gold standard annotations.

Includes:
- CSV/JSON loaders
- Fuzzy matching for headline alignment
- Similarity-based metrics preparation
- Scikit-learn metrics computation

Author: @nicolasleonri (GitHub)
License: GPL
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz
from pathlib import Path
import numpy as np
import difflib
import json
import csv
import re

def parse_ocr_log(file_path):
    """Parse OCR logs using a for loop"""
    print(file_path)
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'File: (.*?) - Config: (.*?) - OCR: (.*?) - Time needed: (.*?) - \[', line.strip())
            # match = re.search(r'File: (.*?) - Config: (.*?) - OCR: (.*?) - Time needed: (.*?) - \[(.*?)\]', line.strip())
            if match:
                # Clean time_needed value - remove trailing 's' if present
                time_str = match.group(4)
                if time_str.endswith('s'):
                    time_str = time_str[:-1]

                entry = {
                    'filename': match.group(1),
                    'config': match.group(2),
                    'ocr': match.group(3),
                    'time_needed': float(time_str)
                }
                results.append(entry)
    
    return results

def parse_preprocessing_log(file_path):
    """Parse preprocessing logs in one line"""
    return [{'filename': m.group(1), 'config': int(m.group(2)), 'time_needed': float(m.group(4))} for line in open(file_path, 'r', encoding='utf-8') if (m := re.search(r'File: (.*?) - Config (\d+): (.*?) - Time needed: (.*?)s', line.strip()))]

def parse_llm_logs_oneliner(file_path):
    """Parse LLM processing logs in one line"""
    return [{'filename': m.group(1), 'config': m.group(2), 'ocr': m.group(3), 'llm': m.group(4), 'time_needed': float(m.group(5))} for line in open(file_path, 'r', encoding='utf-8') if (m := re.search(r'File: (.*?) - Config: (.*?) - OCR: (.*?) - LLM: (.*?) - Time needed: (.*?)s', line.strip()))]


def load_csv_file(file_path: str) -> list:
    """Loads a CSV file into a list of dictionaries."""
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        return list(reader)

def load_json_file(file_path: str) -> dict:
    """Loads a JSON file into a dictionary."""
    with open(file_path, 'r') as file:
        return json.load(file)
    
def find_matching_key(target_key: str, dictionary: dict, threshold: int = 80) -> str:
    """Finds a key in a dictionary that matches a target key approximately."""
    for key in dictionary.keys():
        if fuzz.ratio(target_key.lower(), key.lower()) >= threshold:
            return key
    
    return None

# def preprocess_data(gold_data, eval_data):
#     results = []

#     # Find matching keys for top-level structures
#     articles_key = find_matching_key('articles', eval_data)
#     # brief_news_key = find_matching_key('brief_news', eval_data)

#     if not articles_key:
#         print(f"Warning: Could not find matching keys for articles results.")
#         return None

#     gold_articles = gold_data.get('articles', [])
#     eval_articles = eval_data.get(articles_key, [])

#     used_eval_indices = set()
#     for gold_idx, gold_article in enumerate(gold_articles):
#         gold_headline = gold_article.get("headline", "").lower()
        
#         best_score = 0
#         best_eval_idx = None
        
#         for eval_idx, eval_article in enumerate(eval_articles):
#             if eval_idx in used_eval_indices:
#                 continue
#             eval_headline = eval_article.get("headline", "").lower()
#             score = difflib.SequenceMatcher(None, gold_headline, eval_headline).ratio()
#             if score > best_score:
#                 best_score = score
#                 best_eval_idx = eval_idx
        
#         # If found a close match above threshold
#         if best_eval_idx is not None and best_score > 0.5:
#             used_eval_indices.add(best_eval_idx)
#             eval_article = eval_articles[best_eval_idx]
#         else:
#             eval_article = {}

#         # Compare fields headline, subheadline, content
#         def similarity(a, b):
#             return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

#         results.append({
#             "index": gold_idx,
#             "headline": {
#                 "gold": gold_article.get("headline", ""),
#                 "eval": eval_article.get("headline", ""),
#                 "similarity": similarity(gold_article.get("headline", ""), eval_article.get("headline", ""))
#             },
#             "subheadline": {
#                 "gold": gold_article.get("subheadline", ""),
#                 "eval": eval_article.get("subheadline", ""),
#                 "similarity": similarity(gold_article.get("subheadline", ""), eval_article.get("subheadline", ""))
#             },
#             "content": {
#                 "gold": gold_article.get("content", ""),
#                 "eval": eval_article.get("content", ""),
#                 "similarity": similarity(gold_article.get("content", ""), eval_article.get("content", ""))
#             }
#         })

#     # print(results)

#     return results

def preprocess_data(gold_data: list, eval_data: list) -> list:
    """Aligns and compares articles using fuzzy headline matching.

    Args:
        gold_data (list): List of gold-standard articles.
        eval_data (list): List of evaluated articles.

    Returns:
        list: Similarity data per article field.
    """
    results = []
    used_eval_indices = set()

    for gold_idx, gold_article in enumerate(gold_data):
        gold_headline = gold_article.get("headline", "").lower()

        best_score = 0
        best_eval_idx = None

        for eval_idx, eval_article in enumerate(eval_data):
            if eval_idx in used_eval_indices:
                continue
            eval_headline = eval_article.get("headline", "").lower()
            score = difflib.SequenceMatcher(None, gold_headline, eval_headline).ratio()
            if score > best_score:
                best_score = score
                best_eval_idx = eval_idx

        if best_eval_idx is not None and best_score > 0.5:
            used_eval_indices.add(best_eval_idx)
            matched_eval_article = eval_data[best_eval_idx]
        else:
            matched_eval_article = {}

        def similarity(a, b):
            return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

        results.append({
            "index": gold_idx,
            "headline": {
                "gold": gold_article.get("headline", ""),
                "eval": matched_eval_article.get("headline", ""),
                "similarity": similarity(gold_article.get("headline", ""), matched_eval_article.get("headline", ""))
            },
            "subheadline": {
                "gold": gold_article.get("subheadline", ""),
                "eval": matched_eval_article.get("subheadline", ""),
                "similarity": similarity(gold_article.get("subheadline", ""), matched_eval_article.get("subheadline", ""))
            },
            "content": {
                "gold": gold_article.get("content", ""),
                "eval": matched_eval_article.get("content", ""),
                "similarity": similarity(gold_article.get("content", ""), matched_eval_article.get("content", ""))
            }
        })

    return results

def prepare_for_sklearn_metrics(comparison_results: list, similarity_threshold: float = 0.75):
    """Converts similarity results into binary labels for metric calculation."""
    y_true = []
    y_pred = []

    for item in comparison_results:
        # Each item corresponds to one gold article comparison, with fields headline, subheadline, content
        for field in ['headline', 'subheadline', 'content']:
            similarity = item.get(field, {}).get('similarity', 0)
            correct = similarity >= similarity_threshold
            y_true.append(1)          # Gold data always correct
            y_pred.append(1 if correct else 0)

    return y_true, y_pred

def calculate_metrics(y_true: list, y_pred: list):
    """Computes accuracy, precision, recall, and F1."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1

def process_gold_standards(gold_standard_dir: str) -> dict:
    """Loads all gold standard CSVs into a dictionary keyed by filename stem."""
    gold_standards = {}
    gold_standard_path = Path(gold_standard_dir)
    
    for file_path in gold_standard_path.glob('*_goldstandard.csv'):
        stem = file_path.stem.split('_')[0]  # Get the part before '_goldstandard'
        gold_data = load_csv_file(str(file_path))
        gold_standards[stem] = gold_data
    
    return gold_standards