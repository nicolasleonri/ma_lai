from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import csv
from typing import Union, List, Dict, Any, Optional, Tuple


def aggregate_single_row(all_model_results, i):
    label_votes = []
    score_map = defaultdict(list)

    for model_name, model_data in all_model_results.items():
        label = model_data["headline"]["label"][i]
        score = model_data["headline"]["score"][i]

        if label != "NA" and score != "NA":
            label_votes.append(label)
            score_map[label].append(float(score))

    if label_votes:
        majority_label = Counter(label_votes).most_common(1)[0][0]
        mean_score = sum(score_map[majority_label]) / len(score_map[majority_label])
    else:
        majority_label = "NA"
        mean_score = "NA"

    return {
        "headline": {
            "label": majority_label,
            "score": mean_score
        }
    }

def read_csv_file(file):
    return pd.read_csv(file, sep=";", decimal=".", na_values="NA", quotechar='"', encoding="utf-8")