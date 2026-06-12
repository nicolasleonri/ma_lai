import csv
from pathlib import Path
import subprocess
import torch
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc
import torch

def clear_gpu_memory():
    """Comprehensive GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        print("⚠️ No GPU available, skipping CUDA cleanup")

def process_row(row):
    output_parts = []
    if row.get('headline') != 'NA':
        output_parts.append(row['headline'])
    if row.get('content') != 'NA':
        output_parts.append(row['content'])

    output = ". ".join(output_parts).strip()
    
    if output:
        row['combined_text'] = output
        return output, row
    return None

def process_all_rows(csv_files, max_workers=64):
    print(f"Processing all dataframes with {max_workers} workers")
    all_documents = []
    row_mappings = []
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for csv_file in csv_files:
            if "bertopic" in str(csv_file):
                continue
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    futures.append(executor.submit(process_row, row))

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                output, row = result
                all_documents.append(output)
                row_mappings.append(row)

            if i % 1000 == 0:
                print(f"Processed {i} rows...")

    return all_documents, row_mappings

def get_topics_keywords(model):
    topic_keywords = {}
    topic_info = model.get_topic_info()
    for topic_num in topic_info.Topic:
        if topic_num == -1:  # ignore outliers
            continue
        top_words = model.get_topic(topic_num)
        if top_words:
            label = ", ".join([word for word, _ in top_words[:5]])
            topic_keywords[topic_num] = label
    return topic_keywords

def chunk_documents(documents, rows, chunk_size=100):
    """Yield successive chunks of documents and rows."""
    for i in range(0, len(documents), chunk_size):
        yield documents[i:i + chunk_size], rows[i:i + chunk_size]

def load_csv_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        return list(reader)

def process_csvs(input_folder):
    csvs = {}
    csv_path = Path(input_folder)
    
    for file_path in csv_path.glob('*.csv'):
        stem = str(file_path.stem.split('/'))
        csv_data = load_csv_file(str(file_path))
        csvs[stem] = csv_data

    return csvs

def print_gpu_usage(note=""):
    print(f"\n🔍 GPU Memory Usage {note}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        print(output)
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")

def majority_vote(votes):
    most_common = Counter(votes).most_common(1)
    return most_common[0][0] if most_common else None
