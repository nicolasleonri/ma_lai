from utils_sentiment_analyzer import *
from transformers import pipeline
import multiprocessing as mp
from datasets import Dataset
from datetime import date
import concurrent.futures
from glob import glob
from tqdm import tqdm
import pandas as pd
import argparse
import csv
import os
from typing import Union, List, Dict, Any, Optional, Tuple


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Do sentiment analysis (of newspaper articles' headlines) with majority voting of three spanish specialized models.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--input_folder", type=str, required=True, help="Path to folder containing input csv files")

    return parser.parse_args()

def validate_arguments(args) -> Dict[str, Any]:
    config = {}
    errors = []
    
    if not os.path.exists(args.input_folder):
        errors.append(f"Input folder does not exist: {args.input_folder}")
    else:
        csv_files = list(Path(args.input_folder).rglob("*.csv"))
        if len(csv_files) == 0:
            errors.append(f"No .csv files found in input folder: {args.input_folder}")
        config["csv_files_count"] = len(csv_files)
        config["input_folder"] = args.input_folder
    
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def main():
    args = parse_arguments()
    config = validate_arguments(args)

    results_dir = "./results/csv/sentiment_analysis/"
    os.makedirs(os.path.dirname(results_dir), exist_ok=True)
    csv_filename = f"results_sentiment_analysis_{date.today()}.csv"
    output_csv = os.path.join(results_dir, csv_filename)

    all_files = glob(os.path.join(config["input_folder"], "*.csv"))
    df_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map files to read_csv_file with progress bar
        for df_chunk in tqdm(executor.map(read_csv_file, all_files), total=len(all_files), desc="Reading CSV files"):
            df_list.append(df_chunk)

    df = pd.concat(df_list, ignore_index=True)
    
    print(f"📄 Total valid rows: {len(df)}")

    dataset = Dataset.from_pandas(df)

    models = {
        "sabert": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
        "robertuito": "pysentimiento/robertuito-sentiment-analysis",
        # "lxyuan": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        # "edumunozsala": "edumunozsala/roberta_bne_sentiment_analysis_es",
        "UMUTeam": "UMUTeam/roberta-spanish-sentiment-analysis"
    }

    label_maps = {
        "sabert": {
            "Positive": "Positive",
            "Negative": "Negative",
            "Neutral": "Neutral",
        },
        "lxyuan": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        },
        "robertuito": {
            "POS": "Positive",
            "NEG": "Negative",
            "NEU": "Neutral",
        },
        "edumunozsala": {
            "Positivo": "Positive",
            "Negativo": "Negative",
            "Neutral": "Neutral",
        },
        "UMUTeam": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        },
    }

    all_model_results = {}

    for model_name, model_path in models.items():
        print(f"Loading model: {model_name}")
        classifier = pipeline("text-classification", model=model_path, batch_size=64)

        def classify_batch(batch):
            texts = batch.get("headline", [])
            texts = [text if text is not None and text != "NA" else "" for text in texts]

            preds = classifier(texts, truncation=True)

            labels = [label_maps.get(model_name, {}).get(p["label"], p["label"]) for p in preds]
            scores = [p["score"] for p in preds]

            return {
                "headline_label": labels,
                "headline_score": scores
            }
        
        # Run batch inference on dataset
        dataset = dataset.map(classify_batch, batched=True, batch_size=64)

        all_model_results[model_name] = {
            "headline": {
                "label": dataset["headline_label"],
                "score": dataset["headline_score"]
            }
        }

        del classifier

    for model_name, res in all_model_results.items():
        df[f"{model_name}_headline_label"] = res["headline"]["label"]
        df[f"{model_name}_headline_score"] = res["headline"]["score"]

    print("🗳️ Aggregating majority votes and mean scores across models...")
    num_rows = len(next(iter(all_model_results.values()))["headline"]["label"])

    def aggregate_row(i):
        return aggregate_single_row(all_model_results, i)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(aggregate_row, range(num_rows)), total=num_rows))

    aggregated_results = {i: results[i] for i in range(num_rows)}

    df["agreed_headline_label"] = [aggregated_results[i]["headline"]["label"] for i in range(len(df))]
    df["agreed_headline_score"] = [aggregated_results[i]["headline"]["score"] for i in range(len(df))]

    df.to_csv(
        output_csv,
        index=False,
        header=True,
        encoding="utf-8",
        na_rep='NA',
        sep=';',            # Use semicolon as delimiter
        quotechar='"',      # Force double quotes around strings
        date_format='%d-%M-%Y',  # Format datetime columns consistently
        quoting=csv.QUOTE_ALL, # Ensure all fields are quoted
        decimal='.', 
        errors='strict',
    )

    print(f"✅ Results written to: {output_csv}")

    return None

if __name__ == "__main__":
    main()