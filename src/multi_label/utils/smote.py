from typing import List, Dict, Tuple, Optional, Literal, Any
from utils.data_management import *
from nltk.tokenize import sent_tokenize
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import nltk

def chunk_text_by_tokens(text, tokenizer, max_tokens=512):
    """
    Split a long text into token-safe chunks using sentence boundaries.
    """
    sentences = sent_tokenize(text, language="spanish")
    chunks = []
    current = ""

    for s in sentences:
        test = (current + " " + s).strip()
        token_count = len(tokenizer.tokenize(test))

        if token_count > max_tokens:  
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current = test

    if current:
        chunks.append(current.strip())

    return chunks

def simple_paraphrase_augmentation(
    df: pd.DataFrame,
    label_columns: List[str],
    text_column: str = 'combined_text',
    target_samples_per_class: int = 50,
    max_augmentation_per_sample: int = 5,
    model_name: str = "milyiyo/paraphraser-spanish-t5-small",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_beams: int = 5,
    batch_size: int = 4,
    max_tokens_per_chunk: int = 512,
    temp: float = 1.2,
    target_fraction: float = 0.7,
    verbose: bool = True) -> pd.DataFrame:
    """
    Args:
        df: DataFrame con textos y etiquetas
        label_columns: Lista de columnas de etiquetas
        text_column: Nombre de columna con texto
        target_samples_per_class: Objetivo de muestras por clase
        max_augmentation_per_sample: Máximo de paráfrasis por muestra original
        model_name: Modelo de HuggingFace a usar
        device: 'cuda' o 'cpu'
        num_beams: Beam search width (3-10 recomendado)
        temperature: >1.0 = más diverso, <1.0 = más conservador
        batch_size: Batch size para generación (ajustar según GPU)
        verbose: Mostrar progreso
    
    Returns:
        DataFrame balanceado con sintéticas
    """
    
    if verbose:
        print("="*70)
        print("🚀 SIMPLE PARAPHRASE AUGMENTATION")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        print("="*70)
    
    print(f"📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    
    augmentation_plan = {}
    
    for task in label_columns:
        class_counts = df[task].value_counts().to_dict()
        max_class_count = max(class_counts.values())
        
        for cls, current_count in class_counts.items():
            target_count = int(max_class_count * target_fraction)
            samples_needed = max(0, target_count - current_count)
            
            if samples_needed > 0:
                class_samples_count = current_count
                
                max_synth_per_original = 5  # you can adjust this
                max_total_synth = class_samples_count * max_synth_per_original
                
                samples_needed = min(samples_needed, max_total_synth)
                
                key = (task, cls)
                augmentation_plan[key] = {
                    'current': current_count,
                    'target': target_count,
                    'needed': samples_needed
                }

    
    if verbose:
        print(f"📋 Plan:")
        print(f"{'Task':<20} {'Class':<8} {'Actual':<10} {'Target':<10} {'To generate':<10}")
        print("-" * 70)
        
        for (task, cls), plan in augmentation_plan.items():
            print(f"{task:<20} {cls:<8} {plan['current']:<10} "
                  f"{plan['target']:<10} {plan['needed']:<10}")
        
        total_to_generate = sum(p['needed'] for p in augmentation_plan.values())
        print("-" * 70)
        print(f"Total synthetic samples to be generated: {total_to_generate}")
    
    synthetic_samples = []
    
    progress_bar = tqdm(
        augmentation_plan.items(),
        desc="Generating paraphrases",
        disable=not verbose
    )
    
    for (task, cls), plan in progress_bar:
        class_samples = df[df[task] == cls]
        if len(class_samples) == 0:
            if verbose:
                print(f"⚠️ No samples for {task}={cls} to augment.")
            continue
        
        samples_needed = plan['needed']
        samples_per_original = min(
            max_augmentation_per_sample,
            int(np.ceil(samples_needed / len(class_samples)))
        )
        
        generated_count = 0
        for idx, base_sample in class_samples.iterrows():
            if generated_count >= samples_needed:
                break
            
            original_text = base_sample[text_column]
            chunks = chunk_text_by_tokens(original_text, tokenizer, max_tokens=max_tokens_per_chunk)
            
            for _ in range(samples_per_original):
                if generated_count >= samples_needed:
                    break
                
                paraphrased_chunks = []
                for chunk in chunks:
                    try:
                        inputs = tokenizer(chunk, return_tensors="pt", max_length=max_tokens_per_chunk,
                                           truncation=True, padding=True).to(device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=max_tokens_per_chunk,
                                num_beams=num_beams,
                                temperature=temp,
                                do_sample=True,
                                top_k=50,
                                top_p=0.95,
                                early_stopping=True,
                                no_repeat_ngram_size=3
                            )
                        paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        paraphrased_chunks.append(paraphrase)
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error in chunk paraphrase: {e}")
                        paraphrased_chunks.append(chunk)
                
                synthetic_text = " ".join(paraphrased_chunks).strip()
                if synthetic_text.lower() == original_text.lower():
                    continue
                
                new_sample = base_sample.copy()
                new_sample[text_column] = synthetic_text
                new_sample['synthetic'] = True
                new_sample['synthetic_method'] = 'chunked_paraphrase_t5'
                new_sample['base_sample_id'] = idx
                new_sample['temperature_used'] = temp
                new_sample['num_chunks'] = len(chunks)
                
                synthetic_samples.append(new_sample)
                generated_count += 1
            
            progress_bar.set_postfix({'task': task, 'class': cls, 'generated': generated_count})
    

    if synthetic_samples:
        df_synthetic = pd.DataFrame(synthetic_samples)
        df_original = df.copy()
        df_original['synthetic'] = False
        df_original['synthetic_method'] = None
        df_original['base_sample_id'] = None
        df_balanced = pd.concat([df_original, df_synthetic], ignore_index=True)
    else:
        df_balanced = df.copy()
        df_balanced['synthetic'] = False
    
    del model
    clean()
    
    if verbose:
        print(f"✅ Augmentation completed:")
        print(f"   - Original samples: {len(df)}")
        print(f"   - Generated synthetic samples: {len(synthetic_samples)}")
        print(f"   - Total in balanced dataset: {len(df_balanced)}")
        
        print(f"📊 DFinal distribution by task:")
        for task in label_columns:
            print(f"   {task}:")
            final_dist = df_balanced[task].value_counts().sort_index()
            for cls, count in final_dist.items():
                original_count = (df[task] == cls).sum()
                synthetic_count = count - original_count
                print(f"      Class {cls:2d}: {count:4d} total "
                      f"({original_count:4d} orig + {synthetic_count:4d} synth)")
    
    print("="*70)
    
    return df_balanced


def validate_synthetic_quality(
    df_original: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    text_column: str = 'combined_text',
    sample_size: int = 10) -> Dict:
    """
    Valida la calidad de las muestras sintéticas generadas.
    
    Args:
        df_original: DataFrame original
        df_synthetic: DataFrame con muestras sintéticas
        text_column: Columna de texto
        sample_size: Número de ejemplos a mostrar
    
    Returns:
        Dictionary con métricas de calidad y ejemplos
    """
    print("🔍 VALIDATION OF THE QUALITY OF SYNTHETIC SAMPLES")
    print("="*70)
    
    synthetic_only = df_synthetic[df_synthetic['synthetic'] == True]
    
    print(f"📊 Statistics:")
    print(f"   - Total synthetic samples: {len(synthetic_only)}")
    print(f"   - Employed methods:")
    
    method_counts = synthetic_only['synthetic_method'].value_counts()
    for method, count in method_counts.items():
        print(f"      • {method}: {count} ({count/len(synthetic_only)*100:.1f}%)")
    
    synthetic_lengths = synthetic_only[text_column].str.len()
    original_lengths = df_original[text_column].str.len()
    
    print(f"📏 Text length:")
    print(f"   - Original average: {original_lengths.mean():.0f} characters")
    print(f"   - Synthetic average: {synthetic_lengths.mean():.0f} characters")
    print(f"   - Difference: {abs(synthetic_lengths.mean() - original_lengths.mean()):.0f} characters")
    
    print(f"📝 Examples of synthetic samples (first {sample_size}):")
    print("-" * 70)
    
    for idx, row in synthetic_only.head(sample_size).iterrows():
        base_id = row['base_sample_id']
        original = df_original.loc[base_id, text_column]
        synthetic = row[text_column]
        method = row['synthetic_method']
        
        print(f"Example {idx + 1} - Method: {method}")
        print(f"Original:  {original[:150]}...")
        print(f"Synthetic: {synthetic[:150]}...")
        print("-" * 70)
    
    return {
        'total_synthetic': len(synthetic_only),
        'method_distribution': method_counts.to_dict(),
        'avg_length_original': original_lengths.mean(),
        'avg_length_synthetic': synthetic_lengths.mean()
    }