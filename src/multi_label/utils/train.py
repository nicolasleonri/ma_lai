from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Any, Optional
from torch.utils.data.dataset import Dataset
from utils.config import TrainingConfig
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from utils.data_management import *
import torch.nn.functional as F
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
import torch.nn as nn
import torch
import numpy as np
import gc

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss para clasificación multicategoría.
        alpha: dict {clase: peso} o tensor de pesos.
        gamma: parámetro de enfoque.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, dict):
            # Convertir dict de pesos por clase a tensor ordenado
            max_idx = max(alpha.keys())
            alpha_tensor = torch.zeros(max_idx + 1)
            for cls_idx, val in alpha.items():
                alpha_tensor[cls_idx] = val
            self.alpha = alpha_tensor
        else:
            self.alpha = alpha  # puede ser tensor o None

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        ce_loss = F.nll_loss(log_probs, targets, reduction='none')

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
    
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

        # focal_weight = (1 - probs) ** self.gamma
        # if self.alpha is not None:
        #     alpha = self.alpha.to(inputs.device)
        #     focal_weight = alpha[targets] * focal_weight

        # loss = F.nll_loss(focal_weight * log_probs, targets, reduction=self.reduction)
        # return loss

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
        print("Loaded Longformer model:", config.model_name)
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
            # loss_fct = nn.CrossEntropyLoss()
            losses = []
            focal_cfg = self.config.focal_loss_config
            gamma = focal_cfg.get('gamma', 2.0)

            for i in range(self.config.num_tasks):
                task_name = self.config.label_columns[i]
                task_labels = labels[:, i] + 1  # pasar de [-1,0,1] → [0,1,2]
                task_logits = logits[:, i, :]
                # task_loss = loss_fct(task_logits, task_labels.long())
                # losses.append(task_loss * self.task_weights[i])
                alpha_dict = focal_cfg['alpha_per_task'].get(task_name, None)
                loss_fct = FocalLoss(alpha=alpha_dict, gamma=gamma)

                task_loss = loss_fct(task_logits, task_labels.long())
                losses.append(task_loss * self.task_weights[i])
            
            loss = torch.stack(losses).sum()
        
        return {"loss": loss, "logits": logits}

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
                            trial_id: Optional[int] = None) -> Dict[str, Any]:
    """K-fold cross validation that returns trainers"""

    num_samples = len(train_labels)
    indices = np.arange(num_samples)
    y = [lbl[0] for lbl in train_labels]
    # kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, y)):
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
        
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    final_metrics = aggregate_fold_metrics(fold_metrics)
    final_metrics["pruned"] = False
    print_cross_validation_results(fold_metrics, final_metrics)
    
    return final_metrics

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
        warmup_ratio=config.warmup_ratio, 
        max_grad_norm=config.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.eval_metric,
        greater_is_better=True,
        fp16=config.use_mixed_precision,
        lr_scheduler_type="constant_with_warmup", # For Hyperparameter Tuning consistency
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # dataloader_pin_memory=True,
        remove_unused_columns=False,
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
