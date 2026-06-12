import json
import os
import glob
import pandas as pd
from collections import defaultdict

def extract_label_f1_from_json(json_path):
    """
    Extract per-label F1 macro scores from a JSON results file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract configuration from filename or directory structure
    # Assuming filename pattern contains config info, or you can parse from path
    filename = os.path.basename(json_path)
    
    # Extract overall metrics
    overall_acc = data.get('eval_accuracy', None)
    macro_f1 = data.get('eval_macro_f1', None)
    mean_task_acc = data.get('eval_mean_task_accuracy', None)
    
    # Extract per-label F1 macro (look for keys ending with '_f1_macro')
    label_f1 = {}
    for key, value in data.items():
        if key.endswith('_f1_macro') and key.startswith('eval_'):
            # Extract label name: eval_MODERNIZANTE_f1_macro -> modernizante
            label = key.replace('eval_', '').replace('_f1_macro', '').lower()
            label_f1[label] = value
    
    return {
        'file': filename,
        'overall_acc': overall_acc,
        'macro_f1': macro_f1,
        'mean_task_acc': mean_task_acc,
        'label_f1': label_f1,
        'full_data': data  # optional, for debugging
    }

def process_all_json_results(json_dir_or_pattern):
    """
    Process all JSON files in directory/matching pattern.
    """
    if os.path.isdir(json_dir_or_pattern):
        json_files = glob.glob(os.path.join(json_dir_or_pattern, '*.json'))
    else:
        json_files = glob.glob(json_dir_or_pattern)
    
    all_results = []
    for json_file in json_files:
        try:
            result = extract_label_f1_from_json(json_file)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return all_results

def find_best_models_per_label(all_results):
    """
    Identify best model for each label based on F1 macro.
    """
    label_best = defaultdict(list)  # label -> list of (score, config)
    
    for result in all_results:
        config = result['file']  # or extract more detailed config from filename/path
        for label, f1 in result['label_f1'].items():
            label_best[label].append((f1, config, result['overall_acc'], result['macro_f1']))
    
    # Sort each label's results by F1 score descending
    for label in label_best:
        label_best[label].sort(key=lambda x: x[0], reverse=True)
    
    return label_best

def print_best_per_label(label_best, top_k=3):
    """
    Print top K models for each label.
    """
    for label, results in sorted(label_best.items()):
        print(f"\n{'='*60}")
        print(f"LABEL: {label.upper()}")
        print(f"{'='*60}")
        print(f"Top {top_k} models:")
        for i, (f1, config, overall_acc, macro_f1) in enumerate(results[:top_k], 1):
            print(f"  {i}. F1={f1:.4f}, Overall Acc={overall_acc:.4f}, "
                  f"Macro F1={macro_f1:.4f}")
            print(f"     Config: {config}")

def create_summary_table(all_results):
    """
    Create a pandas DataFrame summarizing results.
    """
    rows = []
    for result in all_results:
        row = {
            'file': result['file'],
            'overall_acc': result['overall_acc'],
            'macro_f1': result['macro_f1'],
            'mean_task_acc': result['mean_task_acc']
        }
        # Add per-label F1 scores
        for label, f1 in result['label_f1'].items():
            row[f'{label}_f1'] = f1
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def export_to_csv(df, output_path='label_f1_results.csv'):
    """
    Export results to CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"\nResults exported to {output_path}")
    return output_path

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == '__main__':
    # Set your JSON files path/pattern
    # Example patterns:
    #   - Directory: './results/'
    #   - Pattern: './results/*.json'
    #   - Pattern: './results/*/eval_results.json'
    
    json_pattern = './results/csv/multi_label/*.json'  # CHANGE THIS TO YOUR PATH
    
    print(f"Looking for JSON files matching: {json_pattern}")
    all_results = process_all_json_results(json_pattern)
    
    if not all_results:
        print("No JSON files found!")
        exit(1)
    
    print(f"\nProcessed {len(all_results)} JSON result files.")
    
    # Find best models per label
    label_best = find_best_models_per_label(all_results)
    
    # Print best models per label
    print_best_per_label(label_best, top_k=3)
    
    # Create summary DataFrame
    df = create_summary_table(all_results)
    
    # Display summary stats
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS (across all models):")
    print('='*60)
    
    # For each label, show min/mean/max F1
    labels = set()
    for result in all_results:
        labels.update(result['label_f1'].keys())
    
    for label in sorted(labels):
        scores = [r['label_f1'].get(label, None) for r in all_results 
                 if label in r['label_f1']]
        scores = [s for s in scores if s is not None]
        if scores:
            print(f"{label.upper():20} | F1: "
                  f"min={min(scores):.4f}, mean={sum(scores)/len(scores):.4f}, "
                  f"max={max(scores):.4f}")
    
    # Export to CSV
    csv_path = export_to_csv(df)
    
    # Optional: Show DataFrame head
    print(f"\nFirst few rows of summary table:")
    print(df.head())