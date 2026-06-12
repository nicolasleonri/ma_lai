from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from collections import Counter
from bertopic import BERTopic
from utils_bertopic import *
import multiprocessing as mp
from hdbscan import HDBSCAN
from datetime import date
from pathlib import Path
import plotly.io as fig
from umap import UMAP
import numpy as np
import argparse
import pickle
import torch
import time
import csv
import re
import os
import gc

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    print("GPU Memory:", result.stdout.strip())

def run_single_model(documents, embedding_model_name, newspaper, base_model_path, load_model=False, chunk_size=1000):
    spanish_stopwords = [
        'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por',
        'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero',
        'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando',
        'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien',
        'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra',
        'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos',
        'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos',
        'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas',
        'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus',
        'ellas', 'nosotras', 'vosostros', 'vosostras', 'os', 'mío', 'mía', 'míos',
        'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas',
        'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros',
        'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis',
        'están', 'esté', 'estés', 'estemos', 'estéis', 'estén'
    ]

    # Define topics and their seed words
    seeded_topics = [
        [
            "economia informal", "sector informal", "empleo informal", "trabajo informal",
            "informalidad", "trabajadores informales", "mercado informal", "actividad informal",
            "subempleo", "precariedad laboral", "evasión", "no registrado", "sin contrato",
            "sin beneficios", "sin protección", "economía sumergida", "economía subterránea",
            "trabajo no regulado", "empleo no declarado", "informal", "informales", "trabajo diario",
            "cachuelo", "Gamarra", "La Parada", "comercio ambulatorio", "mototaxis", "combis",
            "trabajo doméstico", "economía popular", "autoempleo", "trabajadores por cuenta propia", 
            "familiares no remunerados", "políticas de formalización", "economía en la sombra",
            "informalidad persistente", "informalidad", "informal", "mujeres informales"
            "evadir impuestos"
        ], #economia_informal
        [
            "sector dual", "Arthur Lewis", "premoderno", "subsistencia", "industrialización",
            "desarrollo económico", "absorción", "sector capitalista", "etapas de desarrollo",
            "transición", "formalización", "progreso", "modernización", "residuo", "atraso"
        ], #perspectiva_modernizante
        [
            "marxismo", "capitalismo", "exclusión", "salarios bajos", "competencia laboral",
            "explotación", "plusvalía", "reserva de mano de obra", "desigualdad estructural",
            "sistema económico", "clase trabajadora", "precariado", "explotados", "informalidad estructural",
            "dependencia", "periferia", "centro", "migración rural-urbana"
        ], #perspectiva_estructuralista
        [
            "Hernando de Soto", "burocracia", "regulaciones", "impuestos", "intervención estatal",
            "flexibilidad", "autonomía", "libre mercado", "costos de formalización", "trámites",
            "emprendimiento", "libertad económica", "deregulación", "mercado libre", "competitividad",
            "barreras", "racionalidad", "elección individual", "evitar impuestos", "informalidad voluntaria"
        ], #perspectiva_neoliberal
        [
            "redes de solidaridad", "antropología", "cultura", "reciprocidad", "comunidad",
            "capital social", "trueque", "mercados populares", "identidad", "tradición",
            "resistencia cultural", "economía alternativa", "redistribución", "cooperación",
            "valores comunitarios", "informalidad cultural", "prácticas locales", "solidaridad"
        ], #perspectiva_posmoderna
        [
            "evasión", "competencia desleal", "regulaciones ineficientes", "beneficios",
            "maximización", "estrategia", "ventaja", "mercado libre", "abuso de controles",
            "ineficiencia estatal", "opción racional", "beneficio individual", "eludir normas",
            "informalidad estratégica", "rentabilidad"
        ] #perspectiva_voluntarista
    ]
    
    check_gpu()
    start_time = time.time()

    embedding_model = SentenceTransformer(embedding_model_name, 
        model_kwargs={
        "torch_dtype": "float16",
        # "attn_implementation": "flash_attention_2",
        }
        ) # Create fresh embedding model and BERTopic instance
    
    vectorizer_model = CountVectorizer(
        stop_words=spanish_stopwords,
        decode_error="replace",
        ngram_range=(1, 3),
        max_features=None, # alternative: 100000
        strip_accents="unicode",
        min_df=2,  # Remove very rare terms
        max_df=0.95,  # Remove very common terms
        token_pattern=r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{2,}\b'  # Spanish-aware tokenization # Min 2 chars
    )

    umap_model = UMAP(
        n_neighbors=103, #slow: 30 # fast: 175
        n_components=58, #slow: 15 # fast: 100
        min_dist=0.01, #slow: 0.0
        # metric='cosine',
        random_state=42,
        low_memory=False,
        n_jobs=-1,
        verbose=True
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=16, #slow: 6 # fast: 25
        min_samples=5, #slow: 2 # fast: 7
        # metric='cosine',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1  # Use all CPU cores
        # algorithm='boruvka_kdtree'
    )

    model = BERTopic(
        embedding_model=embedding_model, 
        language="multilingual", 
        min_topic_size=4, #slow: 3 # fast: 5
        top_n_words=35, # Words per topic
        seed_topic_list=seeded_topics,
        representation_model=KeyBERTInspired(),
        calculate_probabilities=True,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        low_memory=False,
        nr_topics=None,
    )
    
    print(f"Initializing time: {time.time() - start_time:.2f} seconds")
    check_gpu()

    model_suffix = re.sub(r'\W+', '_', embedding_model_name.split('/')[-1])
    model_path = f"{base_model_path}/bertopic/bertopic_model_{model_suffix}_{newspaper}_{date.today()}"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle_file = model_path + ".pkl"
    loaded_saved_model = str(model_path)

    try:
        if load_model == True:
            if os.path.exists(model_path) and os.path.exists(pickle_file):
                print(f"Loading existing BERTopic model from {model_path}")
                start_time = time.time()
                model = BERTopic.load(model_path)
                with open(pickle_file, "rb") as f:
                    results = pickle.load(f)
                print(f"Loading time: {time.time() - start_time:.2f} seconds")
        else:
            print(f"Training BERTopic model with {embedding_model_name}")
            model.verbose = True

            start_time = time.time()
            topics, probs = model.fit_transform(documents)
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            print(f"Number of topics: {len(set(topics)) - 1}")  # -1 for outlier topic
            topic_keywords = get_topics_keywords(model)
            model.save(model_path, save_embedding_model=True)
            topic_info = model.get_topic_info().to_dict('records') if hasattr(model, 'get_topic_info') else None
            results = {
                'topics': topics,
                'probs': probs,
                'topic_info': topic_info,
                'topic_keywords': topic_keywords
            }
            # print(results)
            # print(topic_info)
            print(topic_keywords)
            # print(f"Total no of topics: {len(set(topics))}")
            print("Topic info (head):")
            topic_info = model.get_topic_info()
            print(topic_info.head())
            print(f"Total no of topics: {len(topic_info)}")

            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
    except Exception as e:
        print(f"Error with {model_suffix}: {e}")
    
    del model
    del embedding_model
    clear_gpu_memory()

    time.sleep(2)
    
    return results, loaded_saved_model


def bertopic(input_files, newspaper, input_folder, all_documents, row_mappings, model_path, load_model=False):
    results_dir = "./results/csv/bertopic/"
    os.makedirs(os.path.dirname(results_dir), exist_ok=True)

    csv_filename = f"results_topics_{newspaper}_{date.today()}.csv"
    output_csv = os.path.join(results_dir, csv_filename)

    print(f"Processing {len(input_files)} CSV files from {str(input_folder)}")

    if not all_documents:
        print("No valid documents found to process.")
        return None

    embedding_model_names = [
        "sentence-transformers/distiluse-base-multilingual-cased-v1",
        # "jaimevera1107/all-MiniLM-L6-v2-similarity-es",
        "hiiamsid/sentence_similarity_spanish_es",
        # "Qwen/Qwen3-Embedding-8B", 
        "Linq-AI-Research/Linq-Embed-Mistral"
    ]
    
    all_model_results = []
    
    saved_models = []

    for model_name in embedding_model_names:
        print(f"\n{'='*50}")
        print(f"Processing with model: {model_name}")
        print(f"{'='*50}")
        
        try:
            results, loaded_saved_model = run_single_model(all_documents, model_name, newspaper, model_path, load_model)
            all_model_results.append(results)
            saved_models.append(str(loaded_saved_model))
            print(f"Completed: {model_name}")
            print(f"Models saved or loaded (so far): {saved_models}")
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            clear_gpu_memory()
            time.sleep(5)
            continue
        
        clear_gpu_memory()
        time.sleep(5)
    
    print(f"Saved models: {saved_models}")

    gc.collect()

    valid_results = [r for r in all_model_results if r is not None]
    
    # if len(valid_results) < 2:
    #     print("❌ Not enough valid model results for majority voting")
    #     return None
    
    print(f"\nProcessing majority vote from {len(valid_results)} models")
    
    topics_by_model = [result['topics'] for result in valid_results]
    probs_by_model = [result['probs'] for result in valid_results]
    model_names_used = [result.get('model_name', f'model_{i}') for i, result in enumerate(valid_results)]
    topic_labels_by_model = [result.get('topic_keywords', {}) for result in valid_results]

    all_rows = []
    for i in range(len(all_documents)):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing document: {i+1}/{len(all_documents)}")
            
        row = row_mappings[i].copy()
        row['combined_text'] = all_documents[i]
        
        # Add individual model results for each document
        for model_idx, model_name in enumerate(model_names_used):
            # Clean model name for column headers (remove special characters)
            clean_model_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_')
            
            # Add topic from this model
            if i < len(topics_by_model[model_idx]):
                model_topic = topics_by_model[model_idx][i]
                row[f'{clean_model_name}_topic'] = model_topic
                model_topic_labels = topic_labels_by_model[model_idx]
                row[f'{clean_model_name}_topic_label'] = model_topic_labels.get(model_topic, "Unknown")
            else:
                row[f'{clean_model_name}_topic'] = -1
                row[f'{clean_model_name}_topic_label'] = "Unknown"
            
            # Add probability from this model
            if i < len(probs_by_model[model_idx]) and probs_by_model[model_idx][i] is not None:
                prob_vector = probs_by_model[model_idx][i]
                if isinstance(prob_vector, (list, tuple, np.ndarray)):
                    # If it's a probability vector, get the max probability
                    max_prob = max(prob_vector) if len(prob_vector) > 0 else 0.0
                    row[f'{clean_model_name}_prob'] = round(max_prob, 4)
                elif isinstance(prob_vector, (int, float)):
                    row[f'{clean_model_name}_prob'] = round(prob_vector, 4)
                else:
                    row[f'{clean_model_name}_prob'] = 0.0
            else:
                row[f'{clean_model_name}_prob'] = 0.0
                
        all_rows.append(row)

    # Write results to CSV
    if all_rows:
        print(f"\nWriting {len(all_rows)} documents to CSV...")
        fieldnames = list(all_rows[0].keys())
        
        with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=fieldnames,
                delimiter=';',
                quotechar='"',
                quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"✅ Results written to: {output_csv}")
        print(f"📊 CSV includes individual results from {len(model_names_used)} models:")
        for model_name in model_names_used:
            clean_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_')
            print(f"   - {clean_name}_topic, {clean_name}_topic_label, {clean_name}_prob")
    else:
        print("❌ No documents found to write")

    return None

    # reference_topics = valid_results[0]['topics']
    # reference_probs = valid_results[0]['probs']
    
    # majority_agreed_topics = []
    # for i in range(len(all_documents)):
    #     if i % 100 == 0:  # Progress indicator
    #         print(f"Processing agreement: {i+1}/{len(all_documents)}")
        
    #     votes = [topics[i] for topics in topics_by_model if i < len(topics)]
        
    #     if len(votes) >= 2 and len(set(votes)) < len(votes):  # At least 2 models agree
    #         agreed_topic = majority_vote(votes)
            
    #         # Calculate mean probability of agreeing models
    #         agreeing_probs = []
    #         for model_idx, vote in enumerate(votes):
    #             if vote == agreed_topic and model_idx < len(probs_by_model):
    #                 if i < len(probs_by_model[model_idx]):
    #                     prob_vector = probs_by_model[model_idx][i]
    #                     if isinstance(prob_vector, (list, tuple, np.ndarray)) and agreed_topic < len(prob_vector):
    #                         topic_prob = prob_vector[agreed_topic]
    #                         if topic_prob is not None:
    #                             agreeing_probs.append(topic_prob)
    #                     elif isinstance(prob_vector, float):
    #                         agreeing_probs.append(prob_vector)
    #                 else:
    #                     agreeing_probs.append(0.5)  # fallback if index out of range
            
    #         mean_agreed_prob = sum(agreeing_probs) / len(agreeing_probs) if agreeing_probs else 0.5
    #         majority_agreed_topics.append((i, agreed_topic, mean_agreed_prob))

    # topic_labels = valid_results[0]['topic_keywords']

    # agreed_rows = []
    # for i, agreed_topic, mean_agreed_prob in majority_agreed_topics:
    #     row = row_mappings[i].copy()
    #     row['combined_text'] = all_documents[i]
    #     # row['topic'] = reference_topics[i] if i < len(reference_topics) else -1
    #     # row['topic_prob'] = reference_probs[i] if i < len(reference_probs) and reference_probs[i] is not None else ""
    #     row['agreed_topic'] = agreed_topic
    #     row['agreed_topic_prob'] = round(mean_agreed_prob, 4)  
    #     row['agreed_topic_label'] = topic_labels.get(agreed_topic, "Unknown")
        
    #     for model_idx, model_name in enumerate(model_names_used):
    #         clean_model_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_')

    #         if i < len(topics_by_model[model_idx]):
    #             model_topic = topics_by_model[model_idx][i]
    #             row[f'{clean_model_name}_topic'] = model_topic
    #             row[f'{clean_model_name}_topic_label'] = topic_labels.get(model_topic, "Unknown")
    #         else:
    #             row[f'{clean_model_name}_topic'] = -1
    #             row[f'{clean_model_name}_topic_label'] = "Unknown"

    #         if i < len(probs_by_model[model_idx]) and probs_by_model[model_idx][i] is not None:
    #             prob_vector = probs_by_model[model_idx][i]
    #             if isinstance(prob_vector, (list, tuple, np.ndarray)):
    #                 # If it's a probability vector, get the max probability
    #                 max_prob = max(prob_vector) if len(prob_vector) > 0 else 0.0
    #                 row[f'{clean_model_name}_prob'] = round(max_prob, 4)
                    
    #                 # Optionally, also store the probability for the agreed topic
    #                 if agreed_topic < len(prob_vector):
    #                     row[f'{clean_model_name}_agreed_topic_prob'] = round(prob_vector[agreed_topic], 4)
    #                 else:
    #                     row[f'{clean_model_name}_agreed_topic_prob'] = 0.0
    #             elif isinstance(prob_vector, (int, float)):
    #                 row[f'{clean_model_name}_prob'] = round(prob_vector, 4)
    #                 row[f'{clean_model_name}_agreed_topic_prob'] = round(prob_vector, 4)
    #             else:
    #                 row[f'{clean_model_name}_prob'] = 0.0
    #                 row[f'{clean_model_name}_agreed_topic_prob'] = 0.0
    #         else:
    #             row[f'{clean_model_name}_prob'] = 0.0
    #             row[f'{clean_model_name}_agreed_topic_prob'] = 0.0
        
    #     agreed_rows.append(row)

    # # Write results to CSV
    # if agreed_rows:
    #     print(f"\nWriting {len(agreed_rows)} agreed topics to CSV...")
    #     fieldnames = list(agreed_rows[0].keys())
        
    #     with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
    #         writer = csv.DictWriter(
    #             f, 
    #             fieldnames=fieldnames,
    #             delimiter=';',
    #             quotechar='"',
    #             quoting=csv.QUOTE_ALL
    #         )
    #         writer.writeheader()
    #         writer.writerows(agreed_rows)
        
    #     print(f"✅ Results written to: {output_csv}")
    #     for model_name in model_names_used:
    #         clean_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_')
    #         print(f"   - {clean_name}_topic, {clean_name}_topic_label, {clean_name}_prob, {clean_name}_agreed_topic_prob")
    # else:
    #     print("❌ No agreed topics found to write")

    # return None

def gec_csv_files(directory) -> List[Path]:
    SUPPORTED_FORMATS = ['.csv']
    
    logs_files: List[Path] = []
    
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            logs_files.append(file)
            
    output: List[Path] = sorted(logs_files)

    return output

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Do semantic clustering (of newspaper articles) using BERTopic with majority voting of three spanish embedding models.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--input_folder", type=str, required=True, help="Path to folder containing input csv files")
    parser.add_argument("-tn", "--type_of_nvidia_card", type=str, required=False, default="H100", choices=["H100", "A5000", "RTX2080", "A100", "V100"], help="Type of NVIDIA GPU card")
    parser.add_argument("-aram", "--available_ram", type=int, required=False, default=9, help="Available system RAM in GB")
    parser.add_argument("-ngpus", "--num_gpus", type=int, required=False, default=1, help="Number of GPUs available for processing")
    parser.add_argument("-vzm", "--visualize_model", type=bool, required=False, default=False, help="Path to a saved BERTopic model to load instead of training")
    parser.add_argument('-n', '--newspaper', type=str, required=True, help='Newspaper name (required)')
    parser.add_argument('-c', '--cpu_num', type=int, required=True, help='Number of available CPUs')
    parser.add_argument('-lm', '--load_model', type=int, required=False, default=False, help='Number of available CPUs')
    parser.add_argument('-mp', '--model_path', type=str, required=True,  help='Path for models')

    return parser.parse_args()

def validate_arguments(args) -> Dict[str, Any]:
    config = {}
    errors = []
    
    if not os.path.exists(args.input_folder):
        errors.append(f"Input folder does not exist: {args.input_folder}")
    else:
        csv_files = list(Path(args.input_folder).rglob("*.csv"))
        if len(csv_files) == 0:
            errors.append(f"No .csv files found in input folder: {args.input_folder}")
        config["csv_files_count"] = len(csv_files)
        config["input_folder"] = args.input_folder
    
    if not args.newspaper:
        errors.append("Newspaper name is required")
    config["newspaper"] = str(args.newspaper)
    
    if args.available_ram <= 0:
        errors.append("Available RAM must be positive")
    if args.num_gpus <= 0:
        errors.append("Number of GPUs must be positive")
    if args.cpu_num <= 0:
        errors.append("Number of CPUs must be positive")
    
    config["gpu_type"] = args.type_of_nvidia_card
    config["available_ram"] = args.available_ram
    config["num_gpus"] = args.num_gpus
    config["cpu_num"] = args.cpu_num
    config["load_model"] = args.load_model
    config["visualize_model"] = args.visualize_model
    config["model_path"] = args.model_path

    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def main() -> None:
    args = parse_arguments()

    config = validate_arguments(args)
    csv_files = gec_csv_files(config["input_folder"])

    all_documents, row_mappings = process_all_rows(csv_files, max_workers=config["cpu_num"])
    print(f"Total valid documents: {len(all_documents)}")

    if config["visualize_model"] == True:
        model = BERTopic.load(config["input_folder"])

        model_id = Path(config["input_folder"]).stem
        model_id = re.sub(r'\W+', '_', model_id)
        
        topic_info = model.get_topic_info()
        print(topic_info.head())

        viz_output_dir = "./results/visualizations/"
        os.makedirs(viz_output_dir, exist_ok=True)

        print("Saving visualizations...")

        #TODO: Fix visualizations
        # model.visualize_documents().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))
        # model.visualize_hierarchy().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))            
        # model.visualize_topics_per_class().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_barchart.html"))
        model.visualize_topics().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))
        model.visualize_barchart(top_n_topics=20).write_html(os.path.join(viz_output_dir, f"{model_id}_topics_barchart.html"))
        model.visualize_heatmap().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_heatmap.html"))
        model.visualize_term_rank().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_termrank.html"))
        
        print(f"✅ Visualizations saved to {viz_output_dir}")
    else:
        start_time = time.time()
        bertopic(csv_files, config["newspaper"], config["input_folder"], all_documents, row_mappings, config["model_path"], config["load_model"])
        print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")
    
    return None
        
if __name__ == "__main__":
    main()