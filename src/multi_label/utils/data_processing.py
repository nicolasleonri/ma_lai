from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, entropy
from utils.config import TrainingConfig
from collections import defaultdict
from typing import Any, Dict, List
from collections import Counter
import pandas as pd
import numpy as np
import argparse
import random
import warnings
import torch
import os
import re

def analyze_class_distribution(df: pd.DataFrame, label_columns: List[str], split_name: str = 'full') -> Dict[str, Any]:
    results = {}
    total_samples = len(df)
    
    for task in label_columns:
        counts = df[task].value_counts().reindex([-1, 0, 1], fill_value=0)
        percentages = (counts / total_samples * 100).round(2)
        
        non_zero_counts = counts[counts > 0]
        imbalance_ratio = non_zero_counts.max() / non_zero_counts.min() if len(non_zero_counts) > 1 else 1
        
        if imbalance_ratio < 3:
            severity = 'balanced'
        elif imbalance_ratio < 10:
            severity = 'moderate'
        else:
            severity = 'severe'
        
        results[task] = {
            'class_counts': counts.to_dict(),
            'class_percentages': percentages.to_dict(),
            'imbalance_ratio': round(imbalance_ratio, 2),
            'severity': severity
        }
    
    imbalance_ratios = [results[task]['imbalance_ratio'] for task in label_columns]
    summary = {
        'split_name': split_name,
        'total_samples': total_samples,
        'mean_imbalance_ratio': round(np.mean(imbalance_ratios), 2),
        'max_imbalance_ratio': round(max(imbalance_ratios), 2),
        'most_imbalanced_task': label_columns[np.argmax(imbalance_ratios)],
        'overall_severity': 'severe' if max(imbalance_ratios) > 10 else 'moderate' if max(imbalance_ratios) > 3 else 'balanced'
    }
    
    results['summary'] = summary
    return results

def analyze_label_combinations(df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Any]:    
    total_samples = len(df)
    combinations = [tuple(row) for row in df[label_columns].values]
    combination_counts = Counter(combinations)
    
    rare_threshold = 5
    rare_combinations = [
        pattern for pattern, count in combination_counts.items()
        if count < rare_threshold or (count / total_samples) < 0.01
    ]
    
    return {
        'total_unique_combinations': len(combination_counts),
        'rare_combinations_count': len(rare_combinations),
        'rare_percentage': round(len(rare_combinations) / len(combination_counts) * 100, 2),
        'most_common_combination': max(combination_counts.items(), key=lambda x: x[1])[0],
        'most_common_count': max(combination_counts.values())
    }

def compare_splits_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Any]:
    issues = []
    
    for task in label_columns:
        train_dist = train_df[task].value_counts(normalize=True).reindex([-1, 0, 1], fill_value=0)
        val_dist = val_df[task].value_counts(normalize=True).reindex([-1, 0, 1], fill_value=0)
        test_dist = test_df[task].value_counts(normalize=True).reindex([-1, 0, 1], fill_value=0)
        
        max_diff = max(
            abs(train_dist - val_dist).max(),
            abs(train_dist - test_dist).max(), 
            abs(val_dist - test_dist).max()
        )
        
        if max_diff > 0.1:
            issues.append(f"{task}: Large distribution shift ({max_diff:.1%})")
        
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            missing_classes = [cls for cls in [-1, 0, 1] if cls not in split_df[task].values]
            if missing_classes:
                issues.append(f"{task}: Classes {missing_classes} missing in {split_name}")
    
    return {
        'consistent': len(issues) == 0,
        'issues': issues,
        'num_issues': len(issues)
    }

def analyze_task_correlations(df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Any]:    
    corr_matrix = df[label_columns].corr()
    
    strong_corrs = []
    for i in range(len(label_columns)):
        for j in range(i + 1, len(label_columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.3:
                strong_corrs.append({
                    'tasks': (label_columns[i], label_columns[j]),
                    'correlation': round(corr, 3)
                })
    
    return {
        'strong_correlations': strong_corrs,
        'has_strong_correlations': len(strong_corrs) > 0
    }

def compute_class_weights(df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Any]:
    weights = {}
    
    for task in label_columns:
        counts = df[task].value_counts().reindex([-1, 0, 1], fill_value=1e-6)
        total = counts.sum()
        
        # Simple inverse frequency weighting
        class_weights = {
            cls: total / (len(counts) * count) 
            for cls, count in counts.items()
        }
        
        weights[task] = class_weights
    
    return weights

def identify_critical_samples(df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Any]:
    label_matrix = df[label_columns].values
    
    all_positive = np.all(label_matrix == 1, axis=1).sum()
    all_negative = np.all(label_matrix == -1, axis=1).sum()
    all_neutral = np.all(label_matrix == 0, axis=1).sum()
    
    warnings = []
    if all_positive < 5:
        warnings.append(f"Only {all_positive} samples with all positive labels")
    if all_negative < 5:
        warnings.append(f"Only {all_negative} samples with all negative labels")
    
    return {
        'all_positive_count': int(all_positive),
        'all_negative_count': int(all_negative), 
        'all_neutral_count': int(all_neutral),
        'warnings': warnings
    }

def recommend_strategies(
    class_distribution: Dict[str, Any],
    label_combinations: Dict[str, Any]) -> Dict[str, Any]:
    
    summary = class_distribution['summary']
    max_ir = summary['max_imbalance_ratio']
    
    # Simple strategy based on max imbalance ratio
    if max_ir < 3:
        strategy = "standard_training"
        techniques = ["No special handling needed"]
    elif max_ir < 10:
        strategy = "class_weights" 
        techniques = ["Use class weights", "Consider moderate oversampling"]
    else:
        strategy = "advanced_imbalance_handling"
        techniques = ["Class weights", "Focal loss", "Aggressive oversampling", "Ensemble methods"]
    
    # Check for rare combinations
    if label_combinations['rare_percentage'] > 50:
        techniques.append("Avoid stratified splitting - use random splits")
    
    return {
        'recommended_strategy': strategy,
        'techniques': techniques,
        'reasoning': f"Maximum imbalance ratio: {max_ir}",
        'most_problematic_task': summary['most_imbalanced_task']
    }

def tokenize_df(config, df, tokenizer):
    texts = df['combined_text'].tolist()
    labels = df[config.label_columns].values.tolist()
    encodings = tokenizer(texts, truncation=True, max_length=4096, padding=True)
    return texts, labels, encodings

def load_and_prepare_data(config: TrainingConfig) -> pd.DataFrame:
    """Load data from CSV and prepare combined text column."""
    df = pd.read_csv(
        config.input_file,
        sep=",", 
        decimal=".", 
        na_values="NA", 
        quotechar='"', 
        encoding="utf-8"
    )
    
    df['combined_text'] = df['headline'].fillna('') + '. ' + df['content'].fillna('')
    
    if config.number_samples is not None:
        df = df.head(config.number_samples)
        print(f"⚠️ Limited to {config.number_samples} samples for testing")
    
    return df

def identify_label_columns(df: pd.DataFrame) -> List[str]:
    """Identify label columns by excluding known non-label columns."""
    not_chosen_columns = ['headline', 'content', 'newspaper', 'date', 'combined_text', 'comentario']
    label_columns = [col for col in df.columns if col not in not_chosen_columns]
    return label_columns

def split_data(df: pd.DataFrame, random_seed: int, test_size: float = 0.15, val_size: float = 0.176):
    """
    Split data into train/val/test sets.
    Default split: 70% train, 15% val, 15% test
    """
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_seed)
    
    print(f"📊 Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df

def validate_arguments(args) -> TrainingConfig:
    """
    Validate command-line arguments and prepare configuration dictionary.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        dict: Validated configuration dictionary
        
    Raises:
        SystemExit: If validation fails
    """
    errors = []

    if not os.path.isfile(args.input_file):
        errors.append(f"Input file does not exist: {args.input_file}")
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)
    if args.train_model == False and args.load_model == False:
        errors.append("Either --train_model or --load_model must be specified")
    elif args.train_model == True and args.load_model == True:
        errors.append("Only one of --train_model or --load_model can be specified")
    if args.model_path == None:
        errors.append("Model path must be specified (--model_path)")
    if args.save_model == False:
        print("Attention! Not saving the model after training.")
    if args.number_samples is not None and args.number_samples <= 0:
        errors.append("Number of samples must be a positive integer")
    elif args.number_samples is not None:
        print(f"Using only the first {args.number_samples} samples for training/evaluation")
    elif args.number_samples is None:
        print("Using all available samples for training/evaluation")

    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    config = TrainingConfig()

    config.input_file = args.input_file
    config.model_path = args.model_path
    config.cross_validate = args.cross_validate
    config.hyperparameter_tuning = args.hyperparameter_tuning
    config.eval_metric = args.eval_metric
    config.type_augmentation = args.type_augmentation
    config.save_model = args.save_model
    config.automatic_augmentation = args.automatic_augmentation
    config.use_pre_trained = args.use_pre_trained
    config.suffix_csv_file = args.suffix_csv_file

    if config.use_pre_trained == True:
        config.model_name = "/scratch/nicolasal97/gec_extractor/longformer-dapt"
        print(f"⚠️ Using re-pretrained model for training: {config.model_name}.")

    if args.number_samples:
        config.number_samples = args.number_samples

    return config

def parse_arguments():
    """
    Parse command-line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Multi-task text classification with Longformer for long Spanish texts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", "--input_file", type=str, required=True, help="Path to input CSV file containing text to annotate or text and labels to train model")
    parser.add_argument('-sm', '--save_model', type=bool, required=False, default=True, help='Flag to save the trained model')
    parser.add_argument('-lm', '--load_model', type=bool, required=False, default=False, help='Flag to load a pretrained model instead of training')
    parser.add_argument('-mp', '--model_path', type=str, required=True, default=None, help='Directory path to save/load the model')
    parser.add_argument('-ns', '--number_samples', type=int, required=False, default=None, help='Limit training to first N samples (for training)')
    parser.add_argument('-tm', '--train_model', type=bool, required=False, default=False, help='Flag to train the model')
    parser.add_argument('-em', '--eval_metric', type=str, required=False, default="accuracy", choices=["accuracy", "macro_f1"], help="Metric to optimize for best model (accuracy or macro_f1)")
    parser.add_argument('-tp', "--training_preset", type=str, required=False, default="default", choices=["small_dataset", "large_dataset", "fine_tuning", "default"], help="Use predefined training configuration preset")
    parser.add_argument('-cv', "--cross_validate", type=bool, required=False, default=False, help="Enable k-fold cross validation")
    parser.add_argument('-ht', "--hyperparameter_tuning", type=bool, required=False, default=False, help="Enable k-fold cross validation")
    parser.add_argument('-ta', "--type_augmentation", type=str, required=False, default=None, choices=["random_deletion", "random_swap", None], help="Probability to apply data augmentation")
    parser.add_argument('-aa', "--automatic_augmentation", type=bool, required=False, default=False, help="Determines if automatic augmentation is applied.")
    parser.add_argument('-up', "--use_pre_trained", type=bool, required=False, default=False, help="Uses re-pretrained model.")
    parser.add_argument('-scf', "--suffix_csv_file", type=str, required=False, default=None, help="Adds suffix to saved csv file.")

    return parser.parse_args()