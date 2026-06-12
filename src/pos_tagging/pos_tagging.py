from typing import Dict, List, Optional, Tuple
from datetime import date
from pathlib import Path
import pandas as pd
import argparse
import spacy_udpipe
import time
import spacy
import json
import glob
import sys
import re
import csv

class POSTaggerAnalyzer:
    def __init__(self):
        # spacy_udpipe.download("es")
        self.udpipe_model = spacy_udpipe.load("es")
        self.nlp = spacy.load("es_dep_news_trf")
        return None

    def preprocess_text(self, text) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r'\s+', ' ', str(text).strip()) # Remove extra whitespace, normalize
        return text
    
    def find_lexems_in_text(self, text: str, lexems: List[str]) -> Optional[str]:
        """Find if any lexem appears at the beginning of words in text"""
        if not text:
            return None
        
        # Split text into words and clean them
        words = re.findall(r'\b\w+\b', text.lower())

        found = []
        for lexem in lexems:
            lexem_lower = lexem.lower()
            if any(word.startswith(lexem_lower) for word in words):
                found.append(lexem)
                
        return found if found else None
    
    def extract_sentences_with_lexem(self, text: str, lexems: List[str]) -> List[str]:
        if not text or not lexems:
            return []
        doc = self.nlp(text)
        matching_sentences = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(lexem.lower() in sent_text for lexem in lexems):
                matching_sentences.append(sent.text.strip())
        
        return matching_sentences
    
    def get_pos_structure_with_deps(self, sentence: str) -> str:
        """Get POS structure with dependency relations, filtering out punctuation"""
        processed = self.udpipe_model(sentence)
        structures = []

        for token in processed:
            upos = token.pos_
            deprel = token.dep_

            if upos not in ['*', 'PUNCT'] and deprel != '*':
                structures.append(f"{upos}({deprel})")
        
        return '-'.join(structures)
    
    def load_pos_codes(self, pos_codes_path: Path) -> Dict[str, str]:
        if pos_codes_path.exists():
            try:
                with open(pos_codes_path, 'r', encoding='utf-8') as f:
                    pos_codes = json.load(f)
                print(f"Loaded {len(pos_codes)} existing POS codes from {pos_codes_path}")
                return pos_codes
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading pos_codes: {e}. Starting with empty dictionary.")
        
        print("Starting with empty POS codes dictionary")
        return {}
    
    def save_pos_codes(self, pos_codes: Dict[str, str], pos_codes_path: Path) -> None:
        """Save pos_codes dictionary to JSON file"""
        try:
            with open(pos_codes_path, 'w', encoding='utf-8') as f:
                json.dump(pos_codes, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(pos_codes)} POS codes to {pos_codes_path}")
        except IOError as e:
            print(f"Error saving pos_codes: {e}")

    def load_lexems(self, lexems_path: str) -> List[str]:
        try:
            with open(lexems_path, 'r', encoding='utf-8') as f:
                lexems = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(lexems)} lexems from {lexems_path}")
            return lexems
        except IOError as e:
            print(f"Error loading lexems from {lexems_path}: {e}")
            return []

    def process_csv_file(self, csv_path: str, lexems: List[str], pos_codes: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        print(f"Processing {csv_path}...")
        
        try:
            # df = pd.read_csv(csv_path, 
            #                 sep=';',
            #                 encoding='utf-8',
            #                 quotechar='"')
            df = pd.read_csv(csv_path,
                            sep=";", 
                            decimal=".", 
                            na_values="NA", 
                            quotechar='"', 
                            encoding="utf-8")
            
            if 'content' not in df.columns:
                print(f"Warning: 'content' column not found in {csv_path}")
                return df, pos_codes
            
            df['content'] = df['content'].apply(self.preprocess_text)

            df_lexems = df.copy()
            df_lexems['found_lexem'] = df_lexems['content'].apply(lambda x: self.find_lexems_in_text(x, lexems))            
            lexem_rows = df_lexems[df_lexems['found_lexem'].notna()].copy()
            
            del df_lexems

            if lexem_rows.empty:
                print(f"No lexems found in {csv_path}")
                return df, pos_codes
            
            print(f"Found {len(lexem_rows)} rows with lexems")
            
            pos_codes_found = []
            new_structures = 0
            
            for idx, row in lexem_rows.iterrows():
                sentences = self.extract_sentences_with_lexem(row['content'], row['found_lexem'])
                
                row_codes = []
                if sentences:
                    for sentence in sentences:
                        pos_structure = self.get_pos_structure_with_deps(sentence)
                        
                        if pos_structure:
                            if pos_structure not in pos_codes.values():
                                new_code = str(len(pos_codes))
                                pos_codes[new_code] = pos_structure
                                print(f"New POS structure: {new_code} -> {pos_structure}")
                                new_structures += 1
                            else:
                                existing_code = [code for code, struct in pos_codes.items() if struct == pos_structure][0]
                                print(f"Found POS structure: {existing_code} -> {pos_structure}")

                            code = next(k for k, v in pos_codes.items() if v == pos_structure)
                            
                            row_codes.append(code)
                
                # store all codes for the row (or None if empty)
                pos_codes_found.append(row_codes if row_codes else None)

            all_codes = sorted({c for codes in pos_codes_found for c in codes})
            one_hot = pd.DataFrame(0, index=lexem_rows.index, columns=[f"pos_{c}" for c in all_codes])

            # for idx, codes in zip(lexem_rows.index, pos_codes_found):
            #     one_hot.loc[idx, [f"pos_{c}" for c in codes]] = 1

            for idx, codes in zip(lexem_rows.index, pos_codes_found):
                if not codes:  # skip None or empty
                    continue

                # Deduplicate codes and make sure the columns exist
                unique_codes = sorted(set(codes))
                valid_cols = [f"pos_{c}" for c in unique_codes if f"pos_{c}" in one_hot.columns]

                if not valid_cols:
                    continue

                one_hot.loc[idx, valid_cols] = 1

            df = pd.concat([df, one_hot], axis=1)
            df[one_hot.columns] = df[one_hot.columns].fillna(0).astype(int)
            
            del one_hot

            print(f"Added {new_structures} new POS structures")
            print(f"Total POS codes: {len(pos_codes)}\n")
            
            return df, pos_codes
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            return pd.DataFrame(), pos_codes
        
    def process_inputs(self, folder_path: str, lexems_path: str) -> None:
        folder = Path(folder_path)
        lexems = Path(lexems_path)
        
        if not folder.exists() or not lexems.exists():
            print(f"Error: Folder {folder_path} or {lexems_path} do not exist")
            return
        
        lexems = self.load_lexems(str(lexems))
        if not lexems:
            print("No lexems loaded. Exiting.")
            return
        
        pos_codes_path = Path("./logs/pos_codes.json")
        pos_codes = self.load_pos_codes(pos_codes_path)
        
        csv_files = [
            f for f in glob.glob(str(folder / "*.csv"))
            if not f.endswith("_pos.csv")
        ]
        
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process \n")
        
        time_start = time.time()

        total_processed = 0
        for csv_file in csv_files:
            try: 
                df, pos_codes = self.process_csv_file(csv_file, lexems, pos_codes)
                
                if not df.empty:
                    # output_path = csv_file.replace('.csv', '_pos.csv')
                    csv_path = Path(csv_file)
                    output_path = Path(str(csv_path).replace('data/csv/', 'results/csv/')).parent / f"results_pos_{date.today()}{csv_path.suffix}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # df.to_csv(output_path, index=False)
                        df.to_csv(
                            output_path,
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
                        print(f"Saved annotated data to {output_path}")
                        total_processed += 1
                    except Exception as e:
                        print(f"  Error saving {output_path}: {e}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

            del df
        
        total_time = time.time() - time_start

        # Save updated pos_codes
        self.save_pos_codes(pos_codes, pos_codes_path)
        
        print(f"\n=== Processing Complete ===")
        print(f"Files processed: {total_processed}/{len(csv_files)}")
        print(f"Total POS patterns discovered: {len(pos_codes)}")
        print(f"Results saved to: {folder_path}")
        print(f"Time needed: {total_time} seconds")
        print(f"Average time per csv: {total_time/len(csv_files)}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="POS-Tagging Pattern Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
        python pos_tagger.py -f /path/to/data/folder -l /path/to/data/lexems.txt
        python pos_tagger.py -f ./data -l /lexems.txt -m spanish-ancora-ud-2.15-241121.udpipe
        python pos_tagger.py -f ./data -l /lexems.txt -s es_core_news_md
        """
    )

    parser.add_argument("-f", "--input_folder", required=True,
                        help="Path to folder containing CSV files")
    parser.add_argument("-l", "--input_lexem",
                        required=True, help="Path to lexems.txt")

    args = parser.parse_args()

    analyzer = POSTaggerAnalyzer()
    analyzer.process_inputs(args.input_folder, args.input_lexem)


if __name__ == "__main__":
    main()
