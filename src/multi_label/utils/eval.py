import pandas as pd
from typing import Any, Dict, List
from utils.config import TrainingConfig
from utils.data_processing import *
from utils.data_management import *
from sklearn.metrics import f1_score
from utils.train import Trainer, TextClassifierDataset
from utils.data_management import clean
import torch

def evaluate_dataset(df: pd.DataFrame, config: TrainingConfig) -> Dict[str, Any]:
    print("📈 DATASET DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print("1️⃣ Analyzing class distribution...")
    class_distributions = analyze_class_distribution(
        df, 
        config.label_columns, 
        split_name='full'
    )
    
    summary = class_distributions['summary']
    print(f"   📊 Global Summary:")
    print(f"   - Total samples: {summary['total_samples']}")
    print(f"   - Overall severity: {summary['overall_severity'].upper()}")
    print(f"   - Average IR: {summary['mean_imbalance_ratio']:.2f}")
    print(f"   - Maximum IR: {summary['max_imbalance_ratio']:.2f}")
    print(f"   - Most imbalanced task: {summary['most_imbalanced_task']}")
    
    print(f"   📋 Details per task:")
    for task in config.label_columns:
        task_data = class_distributions[task]
        counts = task_data['class_counts']
        percentages = task_data['class_percentages']
        print(f"   - {task:20s}: IR={task_data['imbalance_ratio']:5.2f} | "
              f"Distribution: -1({counts[-1]}, {percentages[-1]:.1f}%) "
              f"0({counts[0]}, {percentages[0]:.1f}%) "
              f"1({counts[1]}, {percentages[1]:.1f}%) | "
              f"[{task_data['severity']}]")

    print("2️⃣ Analyzing label combinations...")
    label_combinations = analyze_label_combinations(df, config.label_columns)
    
    print(f"   🔢 Combination statistics:")
    print(f"   - Unique combinations: {label_combinations['total_unique_combinations']}")
    print(f"   - Rare combinations (<5 samples): {label_combinations['rare_combinations_count']} "
          f"({label_combinations['rare_percentage']:.1f}%)")
    print(f"   - Most common combination: {label_combinations['most_common_combination']}")
    print(f"   - Most common count: {label_combinations['most_common_count']}")

    print("3️⃣ Splitting data into train/val/test...")
    train_df, val_df, test_df = split_data(df, random_seed=config.random_seed)  # 70/15/15
    
    print(f"   - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    print("4️⃣ Comparing distributions between splits...")
    split_comparison = compare_splits_distribution(
        train_df, val_df, test_df, config.label_columns
    )
    
    if split_comparison['consistent']:
        print(f"   ✅ Consistent distributions between splits")
    else:
        print(f"   ⚠️  {split_comparison['num_issues']} problems detected:")
        for issue in split_comparison['issues']:
            print(f"     • {issue}")

    print("5️⃣ Analyzing correlations between tasks...")
    task_correlations = analyze_task_correlations(train_df, config.label_columns)
    
    print(f"   🔗 {len(task_correlations['strong_correlations'])} strong correlations detected (|r| > 0.3)")
    
    if task_correlations['strong_correlations']:
        print(f"   Strongest correlations:")
        for corr in task_correlations['strong_correlations'][:3]:
            task1, task2 = corr['tasks']
            print(f"     • {task1} ↔ {task2}: r={corr['correlation']:.3f}")
    
    print(f"   - Independent tasks: {not task_correlations['has_strong_correlations']}")

    print("6️⃣ Calculating class weights for loss function...")
    class_weights = compute_class_weights(train_df, config.label_columns)
    
    print(f"   ⚖️  Weights per task (format: [-1, 0, 1]):")
    for task in config.label_columns:
        weights = class_weights[task]
        print(f"   - {task:20s}: [{weights[-1]:.3f}, {weights[0]:.3f}, {weights[1]:.3f}]")

    print("7️⃣ Identifying critical samples...")
    critical_samples = identify_critical_samples(train_df, config.label_columns)
    
    print(f"   📌 Extreme cases:")
    print(f"   - All positive: {critical_samples['all_positive_count']}")
    print(f"   - All negative: {critical_samples['all_negative_count']}")
    print(f"   - All neutral: {critical_samples['all_neutral_count']}")
    
    if critical_samples['warnings']:
        print(f"   ⚠️  Alerts:")
        for warning in critical_samples['warnings']:
            print(f"     • {warning}")

    print("8️⃣ Generating strategic recommendations...")
    recommendations = recommend_strategies(class_distributions, label_combinations)
    
    print(f"   🎯 Recommended Strategy: {recommendations['recommended_strategy'].upper()}")
    print(f"   - Reasoning: {recommendations['reasoning']}")
    print(f"   - Most problematic task: {recommendations['most_problematic_task']}")
    
    print(f"   💡 Recommended techniques:")
    for technique in recommendations['techniques']:
        print(f"     • {technique}")

    clean()
    print("="*60)
    
    return {
        'class_distributions': class_distributions,
        'class_weights': class_weights,
        'recommendations': recommendations,
        'label_combinations': label_combinations,
        'task_correlations': task_correlations,
        'critical_samples': critical_samples
    }

def print_test_results(test_results: Dict[str, Any]):
    """Print formatted test results"""
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    if 'eval_accuracy' in test_results:
        print(f"Overall Accuracy: {test_results['eval_accuracy']:.4f}")
    
    if 'eval_macro_f1' in test_results:
        print(f"Macro F1: {test_results['eval_macro_f1']:.4f}")
    
    print("\nPer-Task Performance:")
    for key, value in test_results.items():
        if '_accuracy' in key and key != 'eval_accuracy' and key != 'eval_mean_task_accuracy':
            task_name = key.replace('eval_', '').replace('_accuracy', '')
            f1_key = f'eval_{task_name}_f1_macro'
            f1_score = test_results.get(f1_key, 'N/A')
            print(f"  {task_name:20s}: Acc={value:.4f}, F1={f1_score if isinstance(f1_score, str) else f'{f1_score:.4f}'}")
    
    print("="*60)

def evaluate_single(config: TrainingConfig, trainer: Trainer, test_encodings: Dict[str, torch.Tensor], test_labels: List[List[float]]) -> Dict[str, Any]:
    """Evaluate the trained model on the test set"""
        
    test_dataset = TextClassifierDataset(
        test_encodings, test_labels, None, 
        swap_ratio=config.swap_ratio,
        type_augmentation=None,
        deletion_prob=None
    )
    
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    
    return eval_results
