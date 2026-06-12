from utils.config import TrainingConfig
from transformers import AutoTokenizer
from datetime import date, datetime
from utils.train import Trainer
from typing import Any, Dict
import numpy as np
import shutil
import torch
import json
import os
import gc

class NumpyEncoder(json.JSONEncoder):
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

def prepare_serializable_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert results dictionary to a serializable format.
    Removes non-serializable objects like models and trainers.
    """
    serializable_results = {}
    
    for key, value in results.items():
        if key in ['best_trainer', 'best_model']:
            continue
        elif key == 'results' and isinstance(value, dict):
            nested_results = {}
            for nested_key, nested_value in value.items():
                if nested_key == 'fold_metrics' and isinstance(nested_value, list):
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

def create_checkpoint_dir(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"model_{timestamp}"
    checkpoint_dir = os.path.join(base_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def clean_checkpoint_dirs(input_dir: str) -> None:
    for item in os.listdir(input_dir):
        if item.startswith("checkpoint-") or item.startswith("fold_"):
            path = os.path.join(input_dir, item)
            shutil.rmtree(path)
            print(f"  Removed {item}")

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
    model_path = f"{config.model_path}/model_{config.type_augmentation}_{config.automatic_augmentation}_{config.hyperparameter_tuning}_{config.cross_validate}_{config.use_pre_trained}_{config.number_samples}_{config.eval_metric}_{date.today()}"

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

    with open(os.path.join(model_path, 'training_config.json'), 'w') as f:
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
            results_filename = f"test_results_{config.type_augmentation}_{config.automatic_augmentation}_{config.hyperparameter_tuning}_{config.cross_validate}_{config.use_pre_trained}_{config.number_samples}_{config.eval_metric}_{date.today()}.json"
        else:
            results_filename = f"test_results_{config.type_augmentation}_{config.automatic_augmentation}_{config.hyperparameter_tuning}_{config.cross_validate}_{config.use_pre_trained}_total_{config.eval_metric}_{date.today()}.json"
        
        results_path = os.path.join(config.results_dir, results_filename)
        os.makedirs(config.results_dir, exist_ok=True)
        
        serializable_results = prepare_serializable_results(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
        print(f"✅ Results saved to: {results_path}")