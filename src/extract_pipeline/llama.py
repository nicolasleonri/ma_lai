from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from llama_cpp import Llama
from pathlib import Path
from PIL import Image
import base64
import time
import argparse
import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import io

llm = None
prompt = None
log_file = None
flag_gestion = False

def match(choice: str) -> None:
  dict = {
    "gestion": {
      "ch_repo_id": "unsloth/Nanonets-OCR-s-GGUF",
      "ch_filename":"mmproj-BF16.gguf",
      "llm_repo_id":"unsloth/Nanonets-OCR-s-GGUF",
      "llm_filename":"Nanonets-OCR-s-Q5_K_M.gguf",
    },
    "trome": {
      "ch_repo_id": "unsloth/Nanonets-OCR-s-GGUF",
      "ch_filename":"mmproj-BF16.gguf",
      "llm_repo_id":"unsloth/Nanonets-OCR-s-GGUF",
      "llm_filename":"Nanonets-OCR-s-Q5_K_M.gguf",
    },
    "elcomercio": {
      "ch_repo_id": "mradermacher/olmOCR-7B-0725-GGUF",
      "ch_filename":"olmOCR-7B-0725.mmproj-Q8_0.gguf",
      "llm_repo_id":"mradermacher/olmOCR-7B-0725-GGUF",
      "llm_filename":"olmOCR-7B-0725.Q5_K_M.gguf",
    },
    "ojo": {
      "ch_repo_id": "mradermacher/RolmOCR-GGUF",
      "ch_filename":"RolmOCR.mmproj-Q8_0.gguf",
      "llm_repo_id":"mradermacher/RolmOCR-GGUF",
      "llm_filename":"RolmOCR.Q5_K_M.gguf",
    },
    "correo": {
      "ch_repo_id": "mradermacher/RolmOCR-GGUF",
      "ch_filename":"RolmOCR.mmproj-Q8_0.gguf",
      "llm_repo_id":"mradermacher/RolmOCR-GGUF",
      "llm_filename":"RolmOCR.Q5_K_M.gguf",
    },
    "publimetro": {
      "ch_repo_id": "mradermacher/RolmOCR-GGUF",
      "ch_filename":"RolmOCR.mmproj-Q8_0.gguf",
      "llm_repo_id":"mradermacher/RolmOCR-GGUF",
      "llm_filename":"RolmOCR.Q5_K_M.gguf",
    },
    "peru21": {
      "ch_repo_id": "mradermacher/olmOCR-7B-0725-GGUF",
      "ch_filename":"olmOCR-7B-0725.mmproj-Q8_0.gguf",
      "llm_repo_id":"mradermacher/olmOCR-7B-0725-GGUF",
      "llm_filename":"olmOCR-7B-0725.Q5_K_M.gguf",
    },
  }

  match choice:
    case "trome":
      return dict["trome"]
    case "ojo":
      return dict["ojo"]
    case "publimetro":
      return dict["publimetro"]
    case "peru21":
      return dict["peru21"]
    case "elcomercio":
      return dict["elcomercio"]
    case "correo":
      return dict["correo"]
    case "gestion":
      return dict["gestion"]
    case _:
      print("Newspaper not recognized. Available options: trome, ojo, publimetro, peru21, elcomercio, correo, gestion.")
      return None

def shrink_image(image_path, max_dim=1024):
  """
  Shrinks an image so that its largest side <= max_dim
  """
  img = Image.open(image_path)
  w, h = img.size

  if max(w, h) <= max_dim:
      return img  # already small enough

  # compute new dimensions preserving aspect ratio
  if w > h:
    new_w = max_dim
    new_h = int(h * max_dim / w)
  else:
    new_h = max_dim
    new_w = int(w * max_dim / h)

  return img.resize((new_w, new_h), Image.BICUBIC)

def init_worker(shared_prompt, shared_log_file, shared_flag_gestion, shared_ch_repo_id, shared_ch_filename, shared_llm_repo_id, shared_llm_filename, shared_mw):
  """Initialize model once per process."""
  global llm, prompt, log_file, flag_gestion

  chat_handler = Qwen25VLChatHandler.from_pretrained(
    repo_id=str(shared_ch_repo_id),
    filename=str(shared_ch_filename),
  )

  if int(shared_mw) > 1:
    smode = 1
  else:
    smode = 0

  llm = Llama.from_pretrained(
    repo_id=str(shared_llm_repo_id),
    filename=str(shared_llm_filename),
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=16384,
    n_threads=int(os.cpu_count()),
    n_batch=8192,
    n_ubatch=4096,
    use_mmap=True,
    use_mlock=True,
    numa=True,
    split_mode=int(smode),
    flash_attn=True,
    verbose=False
  )

  prompt = shared_prompt
  log_file = shared_log_file
  flag_gestion = shared_flag_gestion

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    print("GPU Memory:", result.stdout.strip())

def encode_image(image_path):
  img = shrink_image(image_path, max_dim=1024)
  img_bytes = io.BytesIO()
  img.save(img_bytes, format="PNG")
  img_input = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
  return img_input

def chat_completion(llm, prompt, img_input):
  response = llm.create_chat_completion(
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_input}"}},
      ]}
    ],
    temperature=0.0,
    max_tokens=4096,
    repeat_penalty=1.125,
    top_p=0.95,
    top_k=40,
    min_p=0.05,
    stream=False,
    seed=42
    )
  extracted_text = response["choices"][0]["message"]["content"]
  return extracted_text

def get_image_files(directory):
  SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']
  image_files = []
  
  for file in Path(directory).rglob('*'):
    if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
      image_files.append(file)

  output = sorted(image_files)
  return output

def get_subdirectories(directory):
  deepest_subdirectories = []
  for path in Path(directory).rglob('*'):
    if path.is_dir():
      has_subdirs = any(child.is_dir() for child in path.iterdir())
      if not has_subdirs:
        deepest_subdirectories.append(path)
  output = sorted(deepest_subdirectories)
  return output

def save_to_csv_log(filename, extracted_text, log_file):
  file_exists = Path(log_file).exists()
  
  with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    
    if not file_exists:
      writer.writerow(['filename', 'extracted_text'])
    
    writer.writerow([filename, extracted_text])

def process_image(file_path):
  """Use preloaded model to process one image."""
  global llm, prompt, log_file, flag_gestion

  check_gpu()
  start_time  = time.time()

  try:
    if flag_gestion == True:
      img_list = get_image_files(file_path)

      all_results = []
      for file_path in img_list:
        img_input = encode_image(file_path)
        extracted_text = chat_completion(llm, prompt, img_input)
        all_results.append(extracted_text)
        os.remove(file_path)
        check_gpu()

      extracted_text = " ".join(all_results)
      new_path = str(Path(img_list[0]).parent) + ".png"
      save_to_csv_log(str(new_path), extracted_text, log_file)
    else:
      img_input = encode_image(file_path)
      extracted_text = chat_completion(llm, prompt, img_input)
      save_to_csv_log(str(file_path), extracted_text, log_file)
      os.remove(file_path)

    check_gpu()
    total_time = time.time() - start_time
    
    del img_input, extracted_text
    gc.collect()

    print(f"Total time: {total_time:.2f} seconds")
    return f"✓ Extracted text from {file_path.name}"
  except Exception as e:
    gc.collect()
    return f"✗ Failed to process {file_path}: {e}" 


def main():
  parser = argparse.ArgumentParser(description='OCR for preprocessed images.')
  parser.add_argument('-n', '--newspaper', required=True, help='Newspaper name (optional)')
  parser.add_argument('-f', '--folder_file', required=True, help='Input folder path (required)')
  parser.add_argument('-mw', '--max_workers', required=True, help='Newspaper name (optional)')

  args = parser.parse_args()
  
# shared_prompt = """
# You are an expert OCR system. Extract ALL text content from this newspaper image with perfect accuracy.

# CRITICAL REQUIREMENTS:
# - Read every single word, number, date, and punctuation mark visible in the image
# - The text is in SPANISH - preserve all Spanish accents, tildes, and special characters (ñ, á, é, í, ó, ú, ü)
# - Preserve the original text layout and structure (headlines, paragraphs, columns)
# - Maintain proper spacing between words and sentences
# - Include ALL content: headlines, subheadings, body text, captions, advertisements, page numbers, dates
# - Handle multiple columns by reading left-to-right, top-to-bottom within each column
# - Preserve special characters, accents, and non-English text exactly as shown
# - Do NOT skip any text, even if partially obscured or small
# - Do NOT add explanations, interpretations, or markdown formatting
# - Do NOT summarize or paraphrase - extract the exact text as written

# Return ONLY the raw extracted text content, preserving the natural reading flow of the newspaper.
# DO NOT REPEAT CONTENT. IF YOU REPEAT CONTENT MORE THAN TWICE, YOU WILL RECEIVE A NEGATIVE GRADE (REINFORCEMENT LEARNING, RL)

# WARNING: If you return anything other than raw text (explanations, apologies, formatting, etc.), 
# the entire OCR pipeline will fail and all downstream processing will be corrupted. 
# Your response must contain ONLY the extracted text - nothing else.

# CONTEXT: This is a test. You are being compared to other VLMs. You have to be quick and good.

# TIP: If something is being repeated more than twice, it is -for sure- an error.
# """

  shared_prompt = """
  CONTEXT: You are an expert OCR system. Extract ALL text from this Spanish newspaper image with perfect accuracy.

  REQUIREMENTS:
  - Extract every word, number, date, punctuation mark visible
  - Preserve Spanish accents and special characters (ñ, á, é, í, ó, ú, ü)
  - Maintain the original text layout and structure (headlines, paragraphs, columns)
  - Read left-to-right, top-to-bottom within each column
  - Include ALL content: headlines, body text and captions. Do NOT focus on ads, page numbers or information that does not have content
  - Return ONLY raw extracted text - no explanations or formatting
  - Do NOT repeat content - if text appears twice, include it only once
  - Do NOT summarize or paraphrase - extract the exact text as written

  TASK: Extract the complete text preserving natural newspaper reading flow.
  
  WARNING: If you return anything other than raw text (explanations, apologies, formatting, etc.), the entire OCR pipeline will fail and all downstream processing will be corrupted. Your response must contain ONLY the extracted text - nothing else.
  WARNING: This is a test. You are being compared to other VLMs. You have to be quick and good.
  """

  log_timestamp = time.strftime("%Y%m%d_%H%M%S")
  shared_log_file = f"./logs/test/ocr_log_{log_timestamp}.csv"

  model_info = match(args.newspaper)
  shared_ch_repo_id = model_info["ch_repo_id"]
  shared_ch_filename = model_info["ch_filename"]
  shared_llm_repo_id = model_info["llm_repo_id"]
  shared_llm_filename = model_info["llm_filename"]

  if args.newspaper == "gestion":
    shared_flag_gestion = True
    img_list = get_subdirectories(str(args.folder_file))
  else:
    shared_flag_gestion = False
    img_list = get_image_files(str(args.folder_file))

  shared_mw = int(args.max_workers)
  print(f"Working with {shared_mw} memories.")

  start_time = time.time()

  with ProcessPoolExecutor(max_workers=shared_mw, initializer=init_worker, initargs=(shared_prompt, shared_log_file, shared_flag_gestion, shared_ch_repo_id, shared_ch_filename, shared_llm_repo_id, shared_llm_filename, shared_mw)) as executor:
    futures = [executor.submit(process_image, f) for f in img_list]
    for future in as_completed(futures):
      try:
        result = future.result()
        print(result)
      except Exception as e:
        print(f"Process failed with error: {e}")

  total_time = time.time() - start_time
  print(f"Processed {len(img_list)} images in {total_time:.2f} seconds")
  print(f"Avg per image: {total_time / len(img_list):.2f} sec")
  print(f"Log saved to {shared_log_file}")

  return None

if __name__ == "__main__":
  main()