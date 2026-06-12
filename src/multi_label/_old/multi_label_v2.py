"""
Multi-Task Text Classification with Longformer

This script trains a multi-task learning model for Spanish text classification using
Longformer architecture. It processes long documents (up to 4096 tokens) and performs
5 simultaneous classification tasks with 3 classes each.

"""
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import KFold
from safetensors.torch import load_file
from sklearn.metrics import f1_score
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

@dataclass
class FoldMedianPruner:
    metric_name: str = "accuracy"
    min_folds_before_prune: int = 1
    margin: float = 0.0
    trial_timeout_s: Optional[int] = None

    # internal state
    fold_histories: Dict[int, List[float]] = field(default_factory=dict)
    completed_trials_by_fold: Dict[int, List[float]] = field(default_factory=dict)
    trial_start_time: Dict[int, float] = field(default_factory=dict)

    def start_trial(self, trial_id: int):
        self.fold_histories[trial_id] = []
        self.trial_start_time[trial_id] = time.time()

    def observe_fold(self, trial_id: int, fold_score: float, fold_idx: int):
        self.fold_histories[trial_id].append(fold_score)
        self.completed_trials_by_fold.setdefault(fold_idx, []).append(fold_score)

    def should_prune(self, trial_id: int, fold_idx: int) -> bool:
        # timeout check
        if self.trial_timeout_s is not None:
            if (time.time() - self.trial_start_time[trial_id]) > self.trial_timeout_s:
                return True

        # not enough info yet
        if fold_idx + 1 < self.min_folds_before_prune:
            return False

        cur_scores = self.fold_histories[trial_id]
        running_mean = float(np.mean(cur_scores))

        # Build a baseline from *other* trials at this same fold index
        population = self.completed_trials_by_fold.get(fold_idx, [])
        population = [s for s in population if s is not None]
        if len(population) == 0:
            return False

        median_baseline = float(np.median(population))
        # prune if clearly below the median (minus slack)
        return running_mean + self.margin < median_baseline

@dataclass
class TrainingConfig:
    # Model Configuration
    model_name: str = 'mrm8488/longformer-base-4096-spanish'
    num_tasks: int = 5
    num_classes: int = 3
    dropout_rate: float = 0.15
    random_seed: int = 42

    # Tokenizer Configuration 
    eos_token_id: int = None  
    pad_token_id: int = None  
    bos_token_id: int = None  

    label_columns: list = None
    
    # Training Configuration
    learning_rate: float = 2e-5
    head_learning_rate: float = 3e-4
    batch_size: int = 16
    num_epochs: int = 10
    warmup_ratio: float = 0.15
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Augmentation Configuration
    swap_ratio: float = 0.2
    deletion_prob: float = 0.2
    
    # Cross-Validation
    cross_validate: bool = False
    n_folds: int = 5
    
    # Hyperparameter Tuning
    hyperparameter_tuning: bool = False
    num_trials: int = 10
    
    # Paths
    results_dir: str = "./results/csv/multi_label"
    model_path: str = "/scratch/nicolasal97/gec_extractor/results/models/multi_label"
    input_file: str = ""
    output_dir: str = "/scratch/nicolasal97/multi_label/checkpoints"
    
    # Evaluation
    eval_metric: str = ""
    type_augmentation: str = ""
    number_samples: int = 0
    
    # Advanced Settings
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 3
    save_model: bool = True


class MultiTaskMetrics:
    def __init__(self, num_tasks: int, task_names: List[str]):
        self.num_tasks = num_tasks
        self.task_names = task_names
        
    def compute_detailed_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predicted_classes = np.argmax(predictions, axis=2)
        true_classes = (labels + 1).astype(int)
        
        metrics = {}
        
        metrics["accuracy"] = float((predicted_classes == true_classes).mean())
        
        task_accuracies = []
        task_f1_scores = []
        
        for i in range(self.num_tasks):
            task_pred = predicted_classes[:, i]
            task_true = true_classes[:, i]
            
            task_acc = float((task_pred == task_true).mean())
            task_accuracies.append(task_acc)
            
            task_f1_macro = float(f1_score(task_true, task_pred, average='macro', zero_division=0))
            task_f1_weighted = float(f1_score(task_true, task_pred, average='weighted', zero_division=0))
            task_f1_scores.append(task_f1_macro)
            
            metrics[f"{self.task_names[i]}_accuracy"] = task_acc
            metrics[f"{self.task_names[i]}_f1_macro"] = task_f1_macro
            metrics[f"{self.task_names[i]}_f1_weighted"] = task_f1_weighted
            
            for class_idx in range(3):
                class_mask = (task_true == class_idx)
                if class_mask.any():
                    class_accuracy = float((task_pred[class_mask] == class_idx).mean())
                    metrics[f"{self.task_names[i]}_class_{class_idx}_acc"] = class_accuracy
        
        metrics["mean_task_accuracy"] = float(np.mean(task_accuracies))
        metrics["macro_f1"] = float(np.mean(task_f1_scores))
        
        try:
            task_correlations = np.zeros((self.num_tasks, self.num_tasks))

            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    a = predicted_classes[:, i]
                    b = predicted_classes[:, j]
                    if np.std(a) > 0 and np.std(b) > 0:
                        task_correlations[i, j] = np.corrcoef(a, b)[0, 1]
                    else:
                        task_correlations[i, j] = 0.0
            
            for i in range(self.num_tasks):
                for j in range(i + 1, self.num_tasks):
                    key = f"corr_{self.task_names[i]}_{self.task_names[j]}"
                    metrics[key] = float(task_correlations[i, j])
        except:
            pass
        
        return metrics

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


class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels, tokenizer, swap_ratio, deletion_prob, type_augmentation):
        self.encodings = encodings
        self.labels = labels
        self.tokenizer = tokenizer
        self.swap_ratio = swap_ratio
        self.deletion_prob = deletion_prob
        self.type_augmentation = type_augmentation

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.type_augmentation == "random_swap":
            item = self._random_swap(item)
        elif self.type_augmentation == "random_deletion":
            item = self._random_deletion(item)

        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item
    
    def _random_swap(self, item):
        """Randomly swap words in the sequence"""
        try:
            input_ids = item["input_ids"]
            seq_len = input_ids.size(0)
            
            if seq_len < 6:
                return item

            # Number of swaps proportional to sequence length
            n_swaps = max(1, int(seq_len * self.swap_ratio))

            # Build random index pairs
            idx = torch.randperm(seq_len)[: 2 * n_swaps].view(-1, 2)

            # Swap in-place
            swapped = input_ids.clone()
            swapped[idx[:, 0]], swapped[idx[:, 1]] = swapped[idx[:, 1]].clone(), swapped[idx[:, 0]].clone()

            item["input_ids"] = swapped
                
        except Exception as e:
            print(f"Random swap failed: {e}")
            pass
        
        return item
    
    def _random_deletion(self, item):
        """Randomly delete words with probability"""
        try:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            seq_len = input_ids.size(0)

            if seq_len < 6:
                return item

            # Random keep/drop decision per token
            keep_mask = torch.rand(seq_len) > self.deletion_prob

            # Always keep at least a few tokens
            if keep_mask.sum() < 5:
                return item

            # Replace dropped tokens with the pad token
            pad_id = self.tokenizer.pad_token_id

            dropped = input_ids.clone()
            dropped[~keep_mask] = pad_id
            item["input_ids"] = dropped

            # Fix attention mask
            new_mask = attention_mask.clone()
            new_mask[~keep_mask] = 0
            item["attention_mask"] = new_mask
            
        except Exception as e:
            print(f"Random deletion failed: {e}")
            pass
        
        return item

class MultiTaskLongformer(nn.Module):
    """
    Multi-Task Learning wrapper for Longformer with separate classification heads.
    
    This model uses a shared Longformer encoder with 5 independent classification
    heads, allowing simultaneous learning of related tasks while sharing
    representations.
    
    Args:
        model_name: HuggingFace model identifier for pretrained Longformer
        num_tasks: Number of classification tasks (default: 5)
        num_classes: Number of classes per task (default: 3)
    """
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Load the pretrained Longformer with its original config
        self.longformer = LongformerModel.from_pretrained(config.model_name)
        self.longformer_config = self.longformer.config  # Use the original model's config
        
        # Enhanced classification heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Linear(self.longformer_config.hidden_size, 512),
                nn.GELU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(512, config.num_classes)
            ) for _ in range(config.num_tasks)
        ])
        
        # Layer normalization and task weighting
        self.layer_norm = nn.LayerNorm(self.longformer_config.hidden_size)
        self.task_weights = nn.Parameter(torch.ones(config.num_tasks))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Mean pooling instead of just CLS token
        hidden_states = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        pooled_output = self.layer_norm(pooled_output)
        
        # Generate logits for all tasks
        logits = torch.stack([classifier(pooled_output) for classifier in self.classifiers], dim=1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for i in range(self.config.num_tasks):
                task_labels = labels[:, i] + 1
                task_logits = logits[:, i, :]
                task_loss = loss_fct(task_logits, task_labels.long())
                losses.append(task_loss * self.task_weights[i])
            
            loss = torch.stack(losses).sum()
        
        return {"loss": loss, "logits": logits}

def print_gpu_usage(note="") -> None:
    print(f"\n🔍 GPU Memory Usage: {note}")
    print(f"Total available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Reserved : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

def clean() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def setup_experiment_tracking(config: TrainingConfig, experiment_name: str):
    """Setup experiment tracking (optional)"""
    try:
        import wandb
        wandb.init(
            project="spanish-multitask-classification",
            name=experiment_name,
            config=config.__dict__
        )
        return True
    except ImportError:
        print("Weights & Biases not available. Continuing without experiment tracking.")
        return False

def create_training_arguments(config: TrainingConfig, fold: Optional[int] = None):
    """Create enhanced training arguments"""
    
    if fold is not None:
        output_dir = f"{config.output_dir}/fold_{fold}"
    else:
        output_dir = config.output_dir
        
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        
        # Evaluation and logging
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        
        # Model saving
        load_best_model_at_end=True,
        metric_for_best_model=config.eval_metric,
        greater_is_better=True,
        save_total_limit=1,
        
        # Advanced settings
        fp16=config.use_mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=None,  # Disable external logging for now
    )

def cross_validate_training(config: TrainingConfig, 
                            train_encodings: Dict[str, list], 
                            train_labels: List[List[float]],
                            type_augmentation: str,
                            pruner: Optional[FoldMedianPruner] = None, 
                            trial_id: Optional[int] = None) -> Dict[str, Any]:
    """K-fold cross validation that returns trainers"""

    num_samples = len(train_labels)
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    fold_metrics = []

    if pruner is not None and trial_id is not None:
        pruner.start_trial(trial_id)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(num_samples))):
        print(f"🎯 Training Fold {fold + 1}/{config.n_folds}")

        train_fold_encodings = {k: [v[i] for i in train_idx] for k, v in train_encodings.items()}
        val_fold_encodings   = {k: [v[i] for i in val_idx] for k, v in train_encodings.items()}
        train_fold_labels = [train_labels[i] for i in train_idx]
        val_fold_labels   = [train_labels[i] for i in val_idx]
        

        # Train on this fold and return trainer
        fold_result = train_single_fold(
            config,
            train_fold_encodings,
            train_fold_labels,
            val_fold_encodings,
            val_fold_labels,
            type_augmentation,
            fold
        )

        fold_metrics.append(fold_result)

        if config.eval_metric == "accuracy":
            fold_score = fold_result["eval_metrics"]["eval_accuracy"]
        elif config.eval_metric == "macro_f1":
            fold_score = fold_result["eval_metrics"]["eval_macro_f1"]
        else:
            raise ValueError(f"Unknown eval_metric: {config.eval_metric}")

        if pruner is not None and trial_id is not None:
            pruner.observe_fold(trial_id, fold_score, fold_idx=fold)
            if pruner.should_prune(trial_id, fold_idx=fold):
                print(f"⚡️ Trial {trial_id} pruned after fold {fold + 1}.")
                # Aggregate what we have so far so the caller can still log/compare
                partial = aggregate_fold_metrics(fold_metrics)
                partial["pruned"] = True
                return partial
        
        clean()
    
    final_metrics = aggregate_fold_metrics(fold_metrics)
    final_metrics["pruned"] = False
    print_cross_validation_results(fold_metrics, final_metrics)
    
    return final_metrics

def evaluate_single(config: TrainingConfig, trainer: Trainer, 
                    test_encodings: Dict[str, torch.Tensor], test_labels: List[List[float]]) -> Dict[str, Any]:
    """Evaluate the trained model on the test set"""
        
    test_dataset = TextClassifierDataset(
        test_encodings, test_labels, None, 
        swap_ratio=config.swap_ratio,
        type_augmentation=None,
        deletion_prob=None
    )
    
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    
    return eval_results

def train_single_fold(config: TrainingConfig, 
                    train_encodings: Dict[str, torch.Tensor], train_labels: List[List[float]],
                    val_encodings: Dict[str, torch.Tensor], val_labels: List[List[float]],
                    type_augmentation: str, fold: int):
    """Train model on a single fold and return trainer"""
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, do_lower_case=True)
    
    # Create datasets
    train_dataset = TextClassifierDataset(
        train_encodings, train_labels, tokenizer, 
        swap_ratio=config.swap_ratio,
        type_augmentation=type_augmentation,
        deletion_prob=config.deletion_prob
    )
    val_dataset = TextClassifierDataset(
        val_encodings, val_labels, tokenizer, 
        swap_ratio=0.0,                       # no augmentation
        type_augmentation=None,                # disable augmentation
        deletion_prob=0.0                      # disable deletion
    )
    
    # Initialize model
    model = MultiTaskLongformer(config)
    
    # Setup metrics
    num_tasks = len(train_labels[0])
    task_names = [f"task_{i}" for i in range(num_tasks)]
    metrics_calculator = MultiTaskMetrics(len(config.label_columns), config.label_columns)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/fold_{fold}",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.eval_metric,
        greater_is_better=True,
        fp16=config.use_mixed_precision,
        report_to=None,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # processing_class=tokenizer,
        compute_metrics=metrics_calculator.compute_detailed_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    # Train
    train_result = trainer.train()
    eval_results = trainer.evaluate()
    
    return {
        'fold': fold,
        'train_loss': train_result.metrics['train_loss'],
        'eval_metrics': eval_results,
        'trainer': trainer,  # Store the trainer for later evaluation
        'model': trainer.model
    }

def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across all folds"""
    aggregated = {
        'mean_overall_accuracy': np.mean([m['eval_metrics']['eval_accuracy'] for m in fold_metrics]),
        'std_overall_accuracy': np.std([m['eval_metrics']['eval_accuracy'] for m in fold_metrics]),
        'mean_macro_f1': np.mean([m['eval_metrics']['eval_macro_f1'] for m in fold_metrics]),
        'std_macro_f1': np.std([m['eval_metrics']['eval_macro_f1'] for m in fold_metrics]),
        'mean_train_loss': np.mean([m['train_loss'] for m in fold_metrics]),
        'fold_metrics': fold_metrics
    }
    return aggregated

def print_cross_validation_results(fold_metrics: List[Dict], final_metrics: Dict):
    """Print comprehensive cross-validation results"""
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    for i, fold in enumerate(fold_metrics):
        print(f"Fold {i+1}: "
              f"Accuracy = {fold['eval_metrics']['eval_accuracy']:.4f}, "
              f"Macro F1 = {fold['eval_metrics']['eval_macro_f1']:.4f}, "
              f"Train Loss = {fold['train_loss']:.4f}")
    
    print("\nAGGREGATED METRICS:")
    print(f"Mean Accuracy: {final_metrics['mean_overall_accuracy']:.4f} ± {final_metrics['std_overall_accuracy']:.4f}")
    print(f"Mean Macro F1:  {final_metrics['mean_macro_f1']:.4f} ± {final_metrics['std_macro_f1']:.4f}")
    print(f"Mean Train Loss: {final_metrics['mean_train_loss']:.4f}")
    print("="*60)

def hyperparameter_tuning(config: TrainingConfig, val_encodings: Dict[str, torch.Tensor], val_labels: List[List[float]]) -> Dict[str, Any]:
    np.random.seed(config.random_seed)

    best_score = -float("inf")
    best_config = None
    trial_results = []
    best_trial = None

    pruner = FoldMedianPruner(
        metric_name=config.eval_metric,
        min_folds_before_prune=max(1, int(0.5 * config.n_folds)),
        margin=0.05,
        trial_timeout_s=None  # e.g., 1800 for 30 min per trial
    )
    
    for trial in range(config.num_trials):
        print(f"🔍 Hyperparameter Trial {trial + 1}/{config.num_trials}")
        
        trial_config = replace(
            config,
            learning_rate=float(10**np.random.uniform(-5, -4)),
            batch_size=int(np.random.choice([2, 4, 8])),
            warmup_ratio=float(np.random.choice([0.05, 0.1, 0.15])),
            weight_decay=float(np.random.choice([0.01, 0.001, 0.0001])),
            dropout_rate=float(np.random.choice([0.1, 0.15, 0.2])),
        )

        effective_batch = 32  # TODO: Try with 16
        trial_config.gradient_accumulation_steps = max(1, effective_batch // trial_config.batch_size)

        try:
            results = cross_validate_training(
                trial_config, 
                val_encodings, 
                val_labels,
                type_augmentation=None,
                pruner=pruner, 
                trial_id=trial + 1
            )
            
            if config.eval_metric == "accuracy":
                score = results['mean_overall_accuracy']
            elif config.eval_metric == "macro_f1":
                score = results['mean_macro_f1']
            
            trial_info = {
                'trial_num': trial + 1,
                'config': trial_config,
                'score': score,
                'results': results,
                'pruned': results.get('pruned', False)
            }
            trial_results.append(trial_info)

            status = "PRUNED" if results.get('pruned', False) else "DONE"
            print(f"   [{status}] Score: {score:.4f}")

            if not trial_info["pruned"]:
                if best_trial is None or score > best_score:
                    best_score = score
                    best_trial = trial_info
                    print(f"   🎯 New best configuration! Score: {best_score:.4f}")
                
        except Exception as e:
            print(f"   ❌ Trial {trial + 1} failed: {e}")
            continue

    if best_trial is None:
        raise ValueError("No successful trials completed!")

    best_config = best_trial['config']
    best_results = best_trial['results']
    
    print("\n" + "="*60)
    print("BEST HYPERPARAMETER CONFIGURATION")
    print("="*60)
    print(f"Trial: {best_trial['trial_num']}")
    print(f"Score (Macro F1): {best_score:.4f}")
    print(f"Learning Rate: {best_config.learning_rate}")
    print(f"Batch Size: {best_config.batch_size}")
    print(f"Warmup Ratio: {best_config.warmup_ratio:.3f}")
    print(f"Weight Decay: {best_config.weight_decay}")
    print("="*60)

    best_fold = max(best_results['fold_metrics'], key=lambda x: x['eval_metrics']['eval_accuracy'])
    best_trainer = best_fold.get('trainer')
    best_model = best_fold.get('model')

    print(f"✅ Best model from trial {best_trial['trial_num']}, fold {best_fold['fold']}")
    print(f"   Fold accuracy: {best_fold['eval_metrics']['eval_accuracy']:.4f}\n")
    
    return {
        'best_config': best_config,
        'best_trainer': best_trainer,
        'best_model': best_model,
        'best_results': best_results,
        'best_score': best_score,
        'trial_results': trial_results,
        'best_trial': best_trial
    }

def save_final_model(config: TrainingConfig, trainer: Trainer, model: torch.nn.Module, results: Dict[str, Any] = None):
    """
    Save the final trained model, tokenizer, configuration, and results.
    
    Args:
        config: Training configuration
        trainer: Best trainer object
        model: Best model object
        label_columns: List of label column names
        results: Optional training results dictionary
    """
    # Create model path
    model_path = f"{config.model_path}/model_{config.type_augmentation}_{config.hyperparameter_tuning}_{config.cross_validate}_{config.number_samples}_{config.eval_metric}_{date.today()}"

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"🗑️  Removed existing folder: {model_path}")

    os.makedirs(model_path, exist_ok=True)

    print(f"\n💾 Saving model to: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, do_lower_case=True)
        tokenizer.save_pretrained(model_path)
        print("✅ Tokenizer saved")
    except Exception as e:
        print(f"⚠️  Could not save tokenizer: {e}")
    
    try:
        trainer.save_model(model_path)
        print("✅ Model weights saved")
    except Exception as e:
        print(f"⚠️  Could not save model weights: {e}")
    
    model_config = {
        'model_name': config.model_name,
        'num_tasks': len(config.label_columns),
        'label_columns': config.label_columns,
        'num_labels': 5,
        'num_classes_per_task': 3,
    }
    
    with open(os.path.join(model_path, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    print("✅ Model config saved")

    # Save training configuration
    with open(os.path.join(model_path, 'training_config.json'), 'w') as f:
        # Convert dataclass to dict, handling non-serializable types
        config_dict = {}
        for key, value in config.__dict__.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        json.dump(config_dict, f, indent=2)
    print("✅ Training config saved")

    if results is not None:
        if config.number_samples is not None:
            results_filename = f"test_results_{config.type_augmentation}_{config.hyperparameter_tuning}_{config.cross_validate}_{config.number_samples}_{config.eval_metric}_{date.today()}.json"
        else:
            results_filename = f"test_results_{config.type_augmentation}_{config.hyperparameter_tuning}_{config.cross_validate}_total_{config.eval_metric}_{date.today()}.json"
        
        results_path = os.path.join(config.results_dir, results_filename)
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Prepare serializable results
        serializable_results = prepare_serializable_results(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
        print(f"✅ Results saved to: {results_path}")
    
    # print(f"\n✅ All files saved successfully to: {model_path}")

def prepare_serializable_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert results dictionary to a serializable format.
    Removes non-serializable objects like models and trainers.
    """
    serializable_results = {}
    
    for key, value in results.items():
        if key in ['best_trainer', 'best_model']:
            # Skip trainer and model objects
            continue
        elif key == 'results' and isinstance(value, dict):
            # Handle nested results dictionary
            nested_results = {}
            for nested_key, nested_value in value.items():
                if nested_key == 'fold_metrics' and isinstance(nested_value, list):
                    # Remove model and trainer from fold metrics
                    cleaned_folds = []
                    for fold_data in nested_value:
                        cleaned_fold = {k: v for k, v in fold_data.items() 
                                      if k not in ['model', 'trainer']}
                        cleaned_folds.append(cleaned_fold)
                    nested_results[nested_key] = cleaned_folds
                elif isinstance(nested_value, (np.ndarray, np.generic)):
                    nested_results[nested_key] = nested_value.tolist() if hasattr(nested_value, 'tolist') else float(nested_value)
                elif isinstance(nested_value, (int, float, str, bool, list, dict, type(None))):
                    nested_results[nested_key] = nested_value
                else:
                    nested_results[nested_key] = str(nested_value)
            serializable_results[key] = nested_results
        elif isinstance(value, (np.ndarray, np.generic)):
            serializable_results[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
        elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_results[key] = value
        elif hasattr(value, '__dict__'):
            serializable_results[key] = value.__dict__
        else:
            serializable_results[key] = str(value)
    
    return serializable_results

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

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
    
    # Combine headline and content
    df['combined_text'] = df.apply(
        lambda row: '. '.join(row[['headline', 'content']].dropna().astype(str)),
        axis=1
    )
    
    # Limit samples if specified
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

def print_test_results(test_results: Dict[str, Any]):
    """Print formatted test results"""
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    if 'eval_accuracy' in test_results:
        print(f"Overall Accuracy: {test_results['eval_accuracy']:.4f}")
    
    if 'eval_macro_f1' in test_results:
        print(f"Macro F1: {test_results['eval_macro_f1']:.4f}")
    
    # Print per-task metrics if available
    print("\nPer-Task Performance:")
    for key, value in test_results.items():
        if '_accuracy' in key and key != 'eval_accuracy' and key != 'eval_mean_task_accuracy':
            task_name = key.replace('eval_', '').replace('_accuracy', '')
            f1_key = f'eval_{task_name}_f1_macro'
            f1_score = test_results.get(f1_key, 'N/A')
            print(f"  {task_name:20s}: Acc={value:.4f}, F1={f1_score if isinstance(f1_score, str) else f'{f1_score:.4f}'}")
    
    print("="*60)

def tokenize_df(config, df, tokenizer):
    texts = df['combined_text'].tolist()
    labels = df[config.label_columns].values.tolist()
    encodings = tokenizer(texts, truncation=True, max_length=4096, padding=True)
    return texts, labels, encodings

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
    
    print("🔄 Pre-tokenizing dataset...")
    train_val_texts, train_labels, train_encodings = tokenize_df(config, train_df, tokenizer)
    val_texts, val_labels, val_encodings = tokenize_df(config, val_df, tokenizer)
    test_texts, test_labels, test_encodings = tokenize_df(config, test_df, tokenizer)
    train_val_texts, train_val_labels, train_val_encodings = tokenize_df(config, train_val_df, tokenizer)
    print("✅ Pre-tokenization complete.")
    print("\n" + "="*60)

    best_trainer = None
    best_model = None

    if config.hyperparameter_tuning:
        print("\n🎯 Starting Hyperparameter Tuning")
        tuning_results = hyperparameter_tuning(
            config, 
            val_encodings, 
            val_labels
        )
        config = tuning_results['best_config']
        print(f"✅ Best hyperparameters found with score: {tuning_results['best_score']:.4f}")
        clean()
    
    if config.cross_validate:
        print("🔄 Pre-tokenizing dataset...")
        print("✅ Pre-tokenization complete.")
        print("\n" + "="*60)

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

    if config.save_model == True:
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

# Predefined configurations for different scenarios
TRAINING_PRESETS = {
    "small_dataset": TrainingConfig(
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        num_epochs=15,
        warmup_ratio=0.2,
        weight_decay=0.01,
        dropout_rate=0.2,   # More regularization
    ),
    "large_dataset": TrainingConfig(
        batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_epochs=8,
        warmup_ratio=0.1,
        weight_decay=0.001,
        dropout_rate=0.1,
    ),
    "fine_tuning": TrainingConfig(
        learning_rate=1e-5,
        head_learning_rate=5e-4,
        batch_size=16,
        num_epochs=5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        dropout_rate=0.1,
    )
}
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
    parser.add_argument('-cv', "--cross_validate", required=False, default=False, help="Enable k-fold cross validation")
    parser.add_argument('-ht', "--hyperparameter_tuning", required=False, default=False, help="Enable k-fold cross validation")
    parser.add_argument('-ta', "--type_augmentation", type=str, required=False, default=None, choices=["random_deletion", "random_swap", None], help="Probability to apply data augmentation")
    return parser.parse_args()

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
    
    if args.training_preset != "default" and args.training_preset in TRAINING_PRESETS:
        config = TRAINING_PRESETS[args.training_preset]
    else:
        config = TrainingConfig()

    config.input_file = args.input_file
    config.model_path = args.model_path
    config.cross_validate = args.cross_validate
    config.hyperparameter_tuning = args.hyperparameter_tuning
    config.eval_metric = args.eval_metric
    config.type_augmentation = args.type_augmentation
    config.save_model = args.save_model
    if args.number_samples:
        config.number_samples = args.number_samples

    return config

def create_checkpoint_dir(base_dir: str, preset_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{preset_name}_{timestamp}"
    checkpoint_dir = os.path.join(base_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def clean_checkpoint_dirs(input_dir: str) -> None:
    for item in os.listdir(input_dir):
        if item.startswith("checkpoint-") or item.startswith("fold_"):
            path = os.path.join(input_dir, item)
            shutil.rmtree(path)
            print(f"  Removed {item}")

def main() -> None:
    args = parse_arguments()
    config = validate_arguments(args)
        
    if args.train_model:
        config.output_dir = create_checkpoint_dir(config.output_dir, args.training_preset)
        print(f"📁 Checkpoint directory: {config.output_dir}")
        train(config)
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

