from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from pprint import pprint
import optuna
from utils.train import MultiTaskLongformer, CustomDataset, MultiTaskMetrics
from sklearn.metrics import accuracy_score, f1_score
import torch

def hyperparameter_optimization(config, train_encodings, train_labels, val_encodings, val_labels, tokenizer):
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    metrics_calculator = MultiTaskMetrics(len(config.label_columns), config.label_columns)

    def objective(trial):
        model = MultiTaskLongformer(config)

        learning_rate     = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        per_device_batch  = trial.suggest_categorical("batch_size", [4, 8, 16])
        warmup_ratio      = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        weight_decay      = trial.suggest_float("weight_decay", 0.0, 0.3)
        dropout           = trial.suggest_float("dropout", 0.0, 0.4)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)  
        gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])

        config.learning_rate = learning_rate
        config.batch_size = per_device_batch
        config.gradient_accumulation_steps = gradient_accumulation_steps
        config.warmup_ratio = warmup_ratio
        config.weight_decay = weight_decay
        config.dropout_rate = dropout
        config.max_grad_norm = max_grad_norm

        training_args = TrainingArguments(
            output_dir=f"{config.output_dir}/optuna_trial",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=per_device_batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            report_to=[],
            disable_tqdm=False,
            load_best_model_at_end=False,
            metric_for_best_model=config.eval_metric,
        )

        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                #processing_class=tokenizer,
                compute_metrics=metrics_calculator.compute_detailed_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
            )

            trainer.train()

            eval_results = trainer.evaluate()

            if config.eval_metric == "accuracy":
                metric_value = eval_results["eval_accuracy"]
            elif config.eval_metric == "macro_f1":
                metric_value = eval_results["eval_macro_f1"]

            del model
            del trainer
            torch.cuda.empty_cache()

            return metric_value
        except Exception as e:
            print(f"Trial failed with error: {e}")
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e).lower():
                print(f"CUDA OOM with batch_size {per_device_batch}, accum_steps {gradient_accumulation_steps}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            else:
                return float('inf')
    
    study = optuna.create_study(
        direction="maximize",
        study_name="transformers_optuna_study",
    )

    print("🎯 Running Optuna hyperparameter search...")
    study.optimize(objective, n_trials=config.num_trials)

    print("\n🏆 Best Hyperparameters:")
    pprint(study.best_params)

    return study.best_params

        

