"""
Multi-Task Text Classification with Longformer

This script trains a multi-task learning model for Spanish text classification using
Longformer architecture. It processes long documents (up to 4096 tokens) and performs
5 simultaneous classification tasks with 3 classes each.

"""
from typing import Union, List, Dict, Any, Optional, Tuple
from safetensors.torch import load_file
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from pprint import pprint
import torch.nn as nn
from tqdm import tqdm
from dataclasses import replace
import shutil
import pandas as pd
import numpy as np
import subprocess
import evaluate
import argparse
import time
import torch
import json
import csv
import sys
import gc
import os
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    FillMaskPipeline,
    DataCollatorWithPadding,
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    LongformerModel,
    Pipeline
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency, entropy
from torch.nn import functional as F

from utils.data_management import *
from utils.data_processing import *
from utils.eval import *
from utils.config import TrainingConfig
from utils.smote import *
from utils.train import *
from utils.hyperparameter_tuning import *


class MultiTaskPipeline(Pipeline):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        if device is None:
            device = next(self.model.parameters()).device
        self.device = torch.device(device)
        self.model.to(self.device)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        return self.tokenizer(
            inputs,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(self.device)

    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        return outputs["logits"]

    def postprocess(self, logits):
        preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        preds = preds - 1
        return {"predictions": preds.tolist()}

def handle_severe_imbalance(train_val_df: pd.DataFrame, config: TrainingConfig, tokenizer: Any, class_distributions: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Handle severe class imbalance through synthetic paraphrasing augmentation.
    
    Args:
        train_val_df: Training + validation DataFrame
        config: Training configuration
        tokenizer: Tokenizer for re-tokenization
        
    Returns:
        Tuple of (balanced_dataframe, analysis_results, tokenized_data)
    """
    
    print("⚠️ Severe imbalance detected. Applying synthetic paraphrasing...")
    
    # Apply paraphrasing augmentation
    train_val_df_balanced = simple_paraphrase_augmentation(
        df=train_val_df,
        label_columns=config.label_columns,
        text_column='combined_text',
        target_samples_per_class=config.target_samples_per_class,
        max_augmentation_per_sample=config.max_augmentation_per_sample,
        model_name="milyiyo/paraphraser-spanish-t5-small",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_beams=config.num_beams,
        temp=config.temperature,
        batch_size=config.batch_size,
        verbose=True
    )
    
    # Validate synthetic data quality
    quality_metrics = validate_synthetic_quality(
        df_original=train_val_df,
        df_synthetic=train_val_df_balanced,
        text_column='combined_text',
        sample_size=3
    )
    
    # Analyze post-augmentation distribution
    analysis_results_post = evaluate_dataset(train_val_df_balanced, config)
    class_distributions_post = analysis_results_post['class_distributions']
    
    # Print improvement summary
    print(f"✅ IMPROVEMENT AFTER AUGMENTATION:")
    print(f"   IR average: {class_distributions['summary']['mean_imbalance_ratio']:.2f} → "
          f"{class_distributions_post['summary']['mean_imbalance_ratio']:.2f}")
    print(f"   Severity: {class_distributions['summary']['overall_severity'].upper()} → "
          f"{class_distributions_post['summary']['overall_severity'].upper()}")
    
    # Re-tokenize the balanced dataset
    print("🔄 Re-tokenizing dataset...")
    train_val_texts, train_val_labels, train_val_encodings = tokenize_df(
        config, train_val_df_balanced, tokenizer
    )
    
    # Clean up memory
    clean()
    
    return train_val_df_balanced, analysis_results_post, {
        'texts': train_val_texts,
        'labels': train_val_labels, 
        'encodings': train_val_encodings
    }

def train(config: TrainingConfig) -> Dict[str, Any]:
    print("🚀 Starting (Enhanced) Training Pipeline")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, do_lower_case=True)
    
    df = load_and_prepare_data(config)
    config.label_columns = identify_label_columns(df)
    config.num_tasks = len(config.label_columns)
    
    print(f"📊 Dataset: {len(df)} samples, {len(config.label_columns)} tasks")
    print(f"🏷️ Tasks: {config.label_columns}")
    train_df, val_df, test_df = split_data(df, random_seed=config.random_seed) # 70/15/15
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"   - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print("="*60)

    print("🔄 Pre-tokenizing dataset...")
    train_texts, train_labels, train_encodings = tokenize_df(config, train_df, tokenizer)
    val_texts, val_labels, val_encodings = tokenize_df(config, val_df, tokenizer)
    test_texts, test_labels, test_encodings = tokenize_df(config, test_df, tokenizer)
    train_val_texts, train_val_labels, train_val_encodings = tokenize_df(config, train_val_df, tokenizer)
    print("✅ Pre-tokenization complete.")
    print("="*60)

    best_trainer = None
    best_model = None

    if config.hyperparameter_tuning is True:
        print("🎯 Starting Hyperparameter Optimization")

        if config.automatic_augmentation is True:
            analysis_results = evaluate_dataset(train_df, config)
            class_distributions = analysis_results['class_distributions']
            class_weights = analysis_results['class_weights']
            recommendations = analysis_results['recommendations']

            if class_distributions['summary']['overall_severity'] in ['high', 'severe']:
                train_df, analysis_results, tokenized_data = handle_severe_imbalance(train_df, config, tokenizer, class_distributions)

                class_distributions = analysis_results['class_distributions']
                class_weights = analysis_results['class_weights']
                recommendations = analysis_results['recommendations']
                
                train_texts = tokenized_data['texts']
                train_labels = tokenized_data['labels']
                train_encodings = tokenized_data['encodings']

        class_weights = compute_class_weights(train_df, config.label_columns)
        config.class_weights_per_task = {}

        for task, weight_dict in class_weights.items():
            adjusted_weights = {
                original_class + 1: weight 
                for original_class, weight in weight_dict.items()
            }
            config.class_weights_per_task[task] = adjusted_weights

        config.focal_loss_config = {
            'gamma': 2.0,
            'alpha_per_task': config.class_weights_per_task,
        }

        best_params = hyperparameter_optimization(
            config,
            train_encodings=train_encodings,
            train_labels=train_labels,
            val_encodings=val_encodings,
            val_labels=val_labels,
            tokenizer=tokenizer,
        )

        print("✅ Best Hyperparameters found:")
        pprint(best_params)
        param_mapping = {
            'per_device_batch': 'batch_size',
            'dropout': 'dropout_rate',
        }

        for param, value in best_params.items():
            config_param = param_mapping.get(param, param)
            setattr(config, config_param, value)

        print("✅ Config after updating:")
        print(f"  learning_rate: {config.learning_rate}")
        print(f"  batch_size: {config.batch_size}")
        print(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
        print(f"  warmup_ratio: {config.warmup_ratio}")
        print(f"  weight_decay: {config.weight_decay}")
        print(f"  dropout_rate: {config.dropout_rate}")  # This should now be correct!
        print(f"  max_grad_norm: {config.max_grad_norm}")
        print("="*60)

    if config.cross_validate is True:
        print("🎯 Starting Cross-Validation Training")
        if config.automatic_augmentation is True:
            analysis_results = evaluate_dataset(train_val_df, config)
            class_distributions = analysis_results['class_distributions']
            class_weights = analysis_results['class_weights']
            recommendations = analysis_results['recommendations']

            if class_distributions['summary']['overall_severity'] in ['high', 'severe']:
                train_val_df, analysis_results, tokenized_data = handle_severe_imbalance(train_val_df, config, tokenizer, class_distributions)

                class_distributions = analysis_results['class_distributions']
                class_weights = analysis_results['class_weights']
                recommendations = analysis_results['recommendations']
                
                train_val_texts = tokenized_data['texts']
                train_val_labels = tokenized_data['labels']
                train_val_encodings = tokenized_data['encodings']

        class_weights = compute_class_weights(train_val_df, config.label_columns)
        config.class_weights_per_task = {}

        for task, weight_dict in class_weights.items():
            adjusted_weights = {
                original_class + 1: weight 
                for original_class, weight in weight_dict.items()
            }
            config.class_weights_per_task[task] = adjusted_weights

        config.focal_loss_config = {
            'gamma': 2.0,
            'alpha_per_task': config.class_weights_per_task,
        }

        final_results = cross_validate_training(
            config, 
            train_val_encodings, 
            train_val_labels, 
            config.type_augmentation
        )

        best_fold = max(final_results['fold_metrics'], key=lambda x: x['eval_metrics']['eval_accuracy'])
        best_trainer = best_fold['trainer']
        print(f"✅ Best fold: {best_fold['fold']} with accuracy: {best_fold['eval_metrics']['eval_accuracy']:.4f}")

        print("\n📊 Evaluating on Test Set")
        test_results = evaluate_single(config, best_trainer, test_encodings, test_labels)
        print_test_results(test_results)
        clean()
    else:
        print("🎯 Starting Standard Training (Single Split)")
        if config.automatic_augmentation is True:
            analysis_results = evaluate_dataset(train_df, config)
            class_distributions = analysis_results['class_distributions']
            class_weights = analysis_results['class_weights']
            recommendations = analysis_results['recommendations']

            if class_distributions['summary']['overall_severity'] in ['high', 'severe']:
                train_df, analysis_results, tokenized_data = handle_severe_imbalance(train_df, config, tokenizer, class_distributions)

                class_distributions = analysis_results['class_distributions']
                class_weights = analysis_results['class_weights']
                recommendations = analysis_results['recommendations']
                
                train_labels = tokenized_data['labels']
                train_encodings = tokenized_data['encodings']

        class_weights = compute_class_weights(train_df, config.label_columns)
        config.class_weights_per_task = {}

        for task, weight_dict in class_weights.items():
            adjusted_weights = {
                original_class + 1: weight 
                for original_class, weight in weight_dict.items()
            }
            config.class_weights_per_task[task] = adjusted_weights

        config.focal_loss_config = {
            'gamma': 2.0,
            'alpha_per_task': config.class_weights_per_task,
        }

        final_results = train_single_fold(
            config, 
            train_encodings,
            train_labels,
            val_encodings,
            val_labels,
            config.type_augmentation, 
            fold=None
        )

        best_trainer = final_results.get('trainer')
        best_model = final_results.get('model')

        print("\n📊 Evaluating on Test Set")
        test_results = evaluate_single(config, best_trainer, test_encodings, test_labels)
        print_test_results(test_results)
        clean()

    if config.save_model is True:
        save_final_model(config, best_trainer, best_model, test_results)

    print("✅ Training completed successfully!")
    clean()
    
    return final_results

def load(config) -> None:
    """
    Load a pretrained model from the specified path.
    
    Args:
        config: Configuration dictionary with model path
    """
    print(f"📦 Loading model from {config.model_path}")
    model = MultiTaskLongformer(config)
    state_dict = load_file(f"{config.model_path}/model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, do_lower_case=True)

    print("✅ Model loaded successfully")

    pipe = MultiTaskPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    print("📂 Loading data...")
    df = pd.read_csv(
        config.input_file,
        sep=";", 
        decimal=".", 
        na_values="NA", 
        quotechar='"', 
        encoding="utf-8"
    )

    print("🤖 Annotating data...")
    all_predictions = []

    for text in tqdm(df['combined_text'], desc="Annotating"):
        preds = pipe(text)["predictions"]  # Returns a list of predictions for 5 tasks
        all_predictions.append(preds)

    # Convert list of lists to separate columns
    num_tasks = len(all_predictions[0])
    task_names = ['modernizante', 'estructuralista', 'neoliberal', 'posmoderna', 'voluntarista'][:num_tasks]

    for i, task in enumerate(task_names):
        df[task] = [pred[i] for pred in all_predictions]

    input_filepath = Path(config.model_path)
    file_name = str(input_filepath.stem) + ".csv"
    file_name = file_name.replace("model_", "results_multi_label_")

    output_filepath = os.path.join("./results/csv/multi_label/", file_name)

    df.to_csv(
        output_filepath,
        index=False,
        header=True,
        encoding="utf-8",
        na_rep='NA',
        sep=';',            # Use semicolon as delimiter
        quotechar='"',      # Force double quotes around strings
        date_format='%d-%m-%Y',  # Format datetime columns consistently
        quoting=csv.QUOTE_ALL, # Ensure all fields are quoted
        decimal='.', 
        errors='strict',
    )
    print(f"✅ Annotated CSV saved to {output_filepath}")

def main() -> None:
    args = parse_arguments()
    config = validate_arguments(args)
        
    if args.train_model:
        config.output_dir = create_checkpoint_dir(config.output_dir)
        print("="*60)
        print(f"📁 Checkpoint directory: {config.output_dir}")
        train(config)
        print("="*60)
        print(f"🧹 Cleaning up checkpoints from {config.output_dir}")
        clean_checkpoint_dirs(config.output_dir)
    elif args.load_model:
        load(config)
    
    print("🎉 Process completed successfully!")
    
if __name__ == "__main__":
    clean()
    print_gpu_usage(note="Before running main()")
    main()
    print_gpu_usage(note="After running main()")
    clean()
    print("\n" + "="*60)

# TODO: 
# 1) Continue pretraining the base model on all the unlabeled newspaper text
# 2) 5 fully independent models
# 3) For each task and each class (-1, 0, +1), find optimal thresholds using dev set
# 4) Run a two-stage model (1) Detect mention (0 vs ±1) (2) Decide polarity (+1 vs -1)