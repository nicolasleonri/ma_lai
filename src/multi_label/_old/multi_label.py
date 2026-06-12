"""
Multi-Task Text Classification with Longformer

This script trains a multi-task learning model for Spanish text classification using
Longformer architecture. It processes long documents (up to 4096 tokens) and performs
5 simultaneous classification tasks with 3 classes each.

"""
# Core ML and Deep Learning
import torch
import torch.nn as nn
import numpy as np

# Transformers and Hugging Face
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

# Data Processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data.dataset import Dataset
from safetensors.torch import load_file

# Utilities
from typing import Union, List, Dict, Any, Optional, Tuple
from datetime import date
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import subprocess
import evaluate
import argparse
import json
import csv
import sys
import gc
import os

class MultiTaskPipeline(Pipeline):
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
        preds = preds - 1  # Map back to [-1,0,1]
        return {"predictions": preds.tolist()}

class TextClassifierDataset(Dataset):
    """
    PyTorch Dataset for multi-task text classification.
    
    Handles tokenized text inputs and multi-label outputs for training
    and evaluation of the Longformer model.
    
    Args:
        encodings: Tokenized text inputs from the tokenizer
        labels: Multi-task labels in shape (n_samples, n_tasks)
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a single sample with all task labels.
        
        Returns:
            dict: Contains input tensors and multi-task labels as float32
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
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
    def __init__(self, model_name, num_tasks=5, num_classes=3):
        super().__init__()

        # Load pretrained Longformer encoder
        self.longformer = LongformerModel.from_pretrained(model_name)

        # Required for HuggingFace pipelines and saving/loading
        self.config = self.longformer.config
        self.config.num_tasks = num_tasks
        self.config.num_labels = num_classes
        
        # Create independent classification heads for each task
        self.classifiers = nn.ModuleList([
            nn.Linear(self.longformer.config.hidden_size, num_classes)
            for _ in range(num_tasks)
        ])

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            labels: Ground truth labels (optional, for training)
            
        Returns:
            dict: Contains 'loss' (if labels provided) and 'logits' for all tasks
        """
        # Get contextualized representations from Longformer
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
        
        # Generate predictions from each task-specific head
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        logits = torch.stack(logits, dim=1)  # Shape: (batch_size, num_tasks, num_classes)
        
        # Calculate loss if labels are provided (training mode)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Calculate loss for each task
            losses = []
            for i in range(len(self.classifiers)):
                # Map labels from [-1, 0, 1] to [0, 1, 2] for CrossEntropyLoss
                task_labels = labels[:, i] + 1  # -1→0, 0→1, 1→2
                task_logits = logits[:, i, :]
                losses.append(loss_fct(task_logits, task_labels.long()))

            # Average loss across all tasks
            loss = torch.stack(losses).mean()
        
        return {"loss": loss, "logits": logits}

def print_gpu_usage(note=""):
    print(f"\n🔍 GPU Memory Usage: {note}")
    print(f"Total avilabe: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Reserved : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    try:
        output = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        # print(output)
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")

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
    parser.add_argument('-eo', '--eval_on', type=str, required=False, default="accuracy", choices=["accuracy", "macro_f1"], help="Metric to optimize for best model (accuracy or macro_f1)")
    return parser.parse_args()

def validate_arguments(args) -> Dict[str, Any]:
    """
    Validate command-line arguments and prepare configuration dictionary.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        dict: Validated configuration dictionary
        
    Raises:
        SystemExit: If validation fails
    """
    config = {}
    errors = []
    
    if not os.path.isfile(args.input_file):
        errors.append(f"Input file does not exist: {args.input_file}")
    else:
        config["input_file"] = args.input_file

    config["load_model"] = args.load_model
    config["train_model"] = args.train_model

    if config["train_model"] == False and config["load_model"] == False:
        errors.append("Either --train_model or --load_model must be specified")
    elif config["train_model"] == True and config["load_model"] == True:
        errors.append("Only one of --train_model or --load_model can be specified")

    config["model_path"] = args.model_path

    if config["model_path"] == None:
        errors.append("Model path must be specified (--model_path)")

    config["save_model"] = args.save_model

    if config["save_model"] == False:
        print("Attention! Not saving the model after training.")

    config["number_samples"] = args.number_samples

    if config["number_samples"] is not None and config["number_samples"] <= 0:
        errors.append("Number of samples must be a positive integer")
    elif config["number_samples"] is not None:
        print(f"Using only the first {config['number_samples']} samples for training/evaluation")
    elif config["number_samples"] is None:
        print("Using all available samples for training/evaluation")

    config['eval_on'] = args.eval_on

    # Exit if any validation errors occurred
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for multi-task classification.
    
    Calculates overall accuracy, per-task accuracy, F1 scores, and aggregated
    metrics across all classification tasks.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from the Trainer
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    predictions, labels = eval_pred

    # Convert logits to class predictions
    predicted_classes = np.argmax(predictions, axis=2)  # Shape: (batch_size, num_tasks)
    true_classes = (labels + 1).astype(int)  # Map [-1, 0, 1] to [0, 1, 2]

    num_tasks = predictions.shape[1]

    # Calculate overall accuracy across all tasks
    accuracy = (predicted_classes == true_classes).mean()
    
    # Calculate per-task metrics
    task_accuracies = []
    task_f1_scores = []

    for i in range(num_tasks):
        # Accuracy for this specific task
        task_acc = (predicted_classes[:, i] == true_classes[:, i]).mean()

        # Macro F1 score for this task (average across classes)
        task_f1 = f1_score(
            true_classes[:, i],
            predicted_classes[:, i],
            average='macro'
        )

        task_accuracies.append(task_acc)
        task_f1_scores.append(task_f1)
    
    overall_f1 = np.mean(task_f1_scores)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(overall_f1),
        "mean_task_accuracy": float(np.mean(task_accuracies)),
    }
    
    # Add per-task metrics dynamically
    for i in range(num_tasks):
        metrics[f"task_{i}_acc"] = float(task_accuracies[i])
        metrics[f"task_{i}_f1"] = float(task_f1_scores[i])
    
    return metrics

def train(config) -> None:
    """
    Main training pipeline for the multi-task Longformer classifier.
    
    Steps:
        1. Load and preprocess data
        2. Split into train/validation/test sets
        3. Tokenize text inputs
        4. Initialize model
        5. Train with early stopping
        6. Evaluate on test set
        7. Save model and results
    
    Args:
        config: Configuration dictionary with all training parameters
    """
    print("📂 Loading data...")
    df = pd.read_csv(
        config["input_file"],
        sep=",",
        doublequote=True,
        encoding="utf-8",
        encoding_errors="strict",
        header=0,
        date_format='%Y-%M-%d'
    )

    if config['number_samples'] is not None:
        df = df.head(config['number_samples'])

    df['date'] = pd.to_datetime(df['date'], format='%Y-%M-%d')
    df['newspaper'] = df['newspaper'].astype('category')

    # Combine headline and content into single text field
    df['combined'] = df.apply(
        lambda row: '. '.join(row[['headline', 'content']].dropna().astype(str)),
        axis=1
    )

    print("✂️ Splitting dataset...")
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42) # 85% train+val, 15% test
    train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42) # 70% train, 15% val (from the 85%)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")

    not_chosen_columns = ['headline', 'content', 'newspaper', 'date', 'combined', 'comentario']
    label_columns = [col for col in df.columns if col not in not_chosen_columns]

    num_tasks = len(label_columns)
    print(f"📊 Detected {num_tasks} classification tasks: {label_columns}")

    # Extract labels for each split
    train_labels = train_df[label_columns].values.tolist()
    val_labels = val_df[label_columns].values.tolist()
    test_labels = test_df[label_columns].values.tolist()

    # Extract text data
    train_texts = train_df['combined'].tolist()
    val_texts = val_df['combined'].tolist()
    test_texts = test_df['combined'].tolist()

    print("🔤 Tokenizing texts...")
    tokenizer = AutoTokenizer.from_pretrained(
        'mrm8488/longformer-base-4096-spanish',
        do_lower_case=True
    )
    
    train_encodings = tokenizer(train_texts, truncation=True, max_length=4096)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=4096)
    test_encodings = tokenizer(test_texts, truncation=True, max_length=4096)

    # Data collator handles dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    val_dataset = TextClassifierDataset(val_encodings, val_labels)
    test_dataset = TextClassifierDataset(test_encodings, test_labels)

    print(f"🤖 Initializing model with {num_tasks} task heads...")

    model = MultiTaskLongformer("mrm8488/longformer-base-4096-spanish")

    if config['number_samples'] is not None and config['number_samples'] < 501: 
        per_device_train_batch_size = 32
        per_device_eval_batch_size = 32
        gradient_accumulation_steps = 2
        dataloader_prefetch_factor = 4
        num_train_epochs = 8
        learning_rate = 3e-5
        weight_decay=0.001
        save_total_limit = 3
        warmup_ratio = 0.2
    elif config['number_samples'] is not None and config['number_samples'] < 1001: 
        per_device_train_batch_size = 24
        per_device_eval_batch_size = 24
        gradient_accumulation_steps = 4
        dataloader_prefetch_factor = 2
        num_train_epochs = 6
        learning_rate = 2e-5
        weight_decay = 0.01
        save_total_limit = 2
        warmup_ratio = 0.1

    training_arguments = TrainingArguments(
        output_dir="./results/checkpoints",
        learning_rate=learning_rate,
        eval_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config['eval_on'],
        greater_is_better=True,
        max_grad_norm=1.0,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        push_to_hub=False,
        fp16_full_eval=False,
        remove_unused_columns=False,
    )

    print("🚀 Starting training...")
    print_gpu_usage(note="Before running train()")

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    print_gpu_usage(note="After running train()")

    print("📊 Evaluating on test set...")

    test_results = trainer.evaluate(test_dataset)
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS:")
    print("="*60)
    pprint(test_results)
    print("="*60 + "\n")

    if config['save_model'] == True:
        if config['number_samples'] is not None:
            model_folder = f"model_{config['number_samples']}_{config['eval_on']}_{date.today()}"
        else:
            model_folder = f"model_total_{config['eval_on']}_{date.today()}"
        model_path = os.path.join(config['model_path'], model_folder)
        os.makedirs(model_path, exist_ok=True)

        print(f"💾 Saving model to {model_path}")

        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"✅ Model saved to {model_path}")

    if config['number_samples'] is not None:
        results_path = os.path.join("./results/csv/multi_label", f"test_results_{config['number_samples']}_{config['eval_on']}_{date.today()}.json")
    else:
        results_path = os.path.join("./results/csv/multi_label", f"test_results_total_{config['eval_on']}_{date.today()}.json")
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
        
    print(f"✅ Results saved to {config['model_path']}")

def load(config) -> None:
    """
    Load a pretrained model from the specified path.
    
    Args:
        config: Configuration dictionary with model path
    """
    print(f"📦 Loading model from {config['model_path']}")
    model = MultiTaskLongformer("mrm8488/longformer-base-4096-spanish", num_tasks=5)
    state_dict = load_file(f"{config['model_path']}/model.safetensors")
    model.load_state_dict(state_dict)

    device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    print("✅ Model loaded successfully")

    pipe = MultiTaskPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    print("📂 Loading data...")
    df = pd.read_csv(
        config["input_file"],
        sep=",",
        encoding="utf-8",
        encoding_errors="strict",
        header=0,
        date_format='%Y-%M-%d'
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

    input_filepath = Path(config['model_path'])
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
        date_format='%d-%M-%Y',  # Format datetime columns consistently
        quoting=csv.QUOTE_ALL, # Ensure all fields are quoted
        decimal='.', 
        errors='strict',
    )
    print(f"✅ Annotated CSV saved to {output_filepath}")


def main() -> None:
    args = parse_arguments()
    config = validate_arguments(args)

    if config['train_model']:
        train(config)
    elif config['load_model']:
        load(config)
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print_gpu_usage(note="Before running main()")

    main()

    print_gpu_usage(note="After running main()")

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

