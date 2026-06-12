import os
import glob
from datasets import Dataset
from transformers import (
    LongformerForMaskedLM,
    LongformerTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from charset_normalizer import from_path
import gc
import torch
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from concurrent.futures import ThreadPoolExecutor, as_completed


MODEL_NAME = "allenai/longformer-base-4096"
CORPUS_PATH = "./data/txt/**/**/**/*.txt"
OUTPUT_DIR = "/scratch/nicolasal97/gec_extractor/longformer-dapt"

BLOCK_SIZE = 1024
STRIDE = 768   # 25% overlap

def get_gpus():
    print(torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    return None

def clean_gpu():
    print("🧹 Clearing GPU cache before training...")

    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print("⚠️ Could not clear GPU cache:", e)
        pass
    return None

def subsample_by_token_budget(
    dataset: Dataset,
    tokenizer,
    target_tokens: int
    ):
    """
    Select documents using a hybrid strategy:
    - 50% of the token budget comes from the longest documents
    - 50% comes from the rest (mixed lengths)
    - Selection is randomized within each group
    """

    def select_from_group(indices, budget):
        total = 0
        selected = []
        for i in indices:
            if total + token_lens[i] > budget:
                break
            selected.append(int(i))
            total += token_lens[i]
        return selected, total

    print(f"🔍 Calculating token lengths for {len(dataset)} documents...")

    token_lens = []

    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        tokens = tokenizer(
            row["text"],
            truncation=False,
            add_special_tokens=False
        )
        token_lens.append(len(tokens["input_ids"]))
    
    token_lens = np.array(token_lens)

    sorted_indices = np.argsort(token_lens)[::-1]
    top_half = sorted_indices[:len(sorted_indices)//2]
    bottom_half = sorted_indices[len(sorted_indices)//2:]

    np.random.seed(42)
    np.random.shuffle(top_half)
    np.random.shuffle(bottom_half)

    target_per_group = target_tokens // 2

    top_selected, top_total = select_from_group(top_half, target_per_group)
    bottom_selected, bottom_total = select_from_group(bottom_half, target_per_group)

    selected_indices = top_selected + bottom_selected
    total_tokens = top_total + bottom_total

    print(f"📦 Selected {len(selected_indices)} documents")
    print(f"🔢 Total tokens ≈ {total_tokens:,d} / target {target_tokens:,d}")
    
    return dataset.select(selected_indices)

def safe_read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        result = from_path(path).best()
        if result is None:
            print(f"❌ Could not detect encoding for {path}. Forcing utf-8 replace.")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        text = str(result)
        # print(f"🔧 Auto-decoded {path} using {result.encoding}")
        return text

def load_corpus():
    files = glob.glob(CORPUS_PATH, recursive=True)

    if len(files) == 0:
        raise ValueError("❌ No .txt files found. Check CORPUS_PATH.")

    print(f"📁 Found {len(files)} text files")
    
    texts = []
    done = 0
    for f in files:
        done += 1
        if done % 5000 == 0:
            print(f"📂 Loading {done} / {len(files)}", end="\r")
        try:
            texts.append(safe_read(f))
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
            continue

    print(f"📚 Loaded {len(texts)} documents")

    if len(texts) == 0:
        raise ValueError("❌ All loaded texts are empty.")

    print("🔍 Sample document:")
    print(texts[0][:500])

    return Dataset.from_dict({"text": texts})


def sliding_window_chunk(batch):
    """
    Convert variable-length tokenized docs into overlapping chunks of BLOCK_SIZE tokens.
    Applies the same chunking to input_ids, attention_mask, and labels (if present).
    """
    new_input_ids = []
    new_attention_mask = []
    new_labels = []

    for i in range(len(batch["input_ids"])):
        input_ids = batch["input_ids"][i]
        attention_mask = batch["attention_mask"][i]
        labels = batch.get("labels", [None]*len(batch["input_ids"]))[i]

        length = len(input_ids)

        for start in range(0, length, STRIDE):
            end = start + BLOCK_SIZE
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]

            # Pad final chunk if needed
            pad_len = BLOCK_SIZE - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = chunk_ids + [0] * pad_len
                chunk_mask = chunk_mask + [0] * pad_len

            new_input_ids.append(chunk_ids)
            new_attention_mask.append(chunk_mask)

            if labels is not None:
                new_labels.append(labels)  # replicate label for all chunks

            if end >= length:
                break

    out = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}

    if new_labels:
        out["labels"] = new_labels

    return out

def filter_sequences_with_mask(batch):
    new_batch = {k: [] for k in batch.keys()}
    for i in range(len(batch["input_ids"])):
        labels = batch["labels"][i]
        if any(l != -100 for l in labels):
            for k in batch.keys():
                new_batch[k].append(batch[k][i])
    return new_batch

def main():
    tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)
    model = LongformerForMaskedLM.from_pretrained(MODEL_NAME)

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            model.gradient_checkpointing_enable()
            print("✅ gradient_checkpointing enabled (dist initialized)")
        else:
            print("ℹ️ Distributed not initialized yet; skipping model.gradient_checkpointing_enable()")
    except Exception as e:
        print("⚠️ Could not change gradient checkpointing state:", e)

    model.config.use_cache = False

    print("📚 Loading corpus...")
    dataset = load_corpus()

    print("✂️ Subsampling corpus to target token budget...")
    dataset = subsample_by_token_budget(
        dataset,
        tokenizer,
        target_tokens=3000000000 # TODO: Test with 5 times (rn, needs 6hours) 
    )

    print("✂️ Tokenizing corpus...")
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=False, return_attention_mask=True),
        batched=True,
        batch_size=100,
        num_proc=min(6, max(1, os.cpu_count() // 2)),
        remove_columns=[],
    )

    print("🧱 Applying sliding window chunking...")
    chunked = tokenized.map(
        sliding_window_chunk,
        batched=True,
        batch_size=100,
        num_proc=min(6, max(1, os.cpu_count() // 2)),
        remove_columns=["text"],  
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    print("🧱 Applying flattening...")
    flat = chunked.flatten_indices()
    flat = flat.shuffle()

    print(f"🧪 Final number of MLM sequences: {len(flat)}")
    print("🔍 Example sequence length:", len(flat[0]["input_ids"]))

    if len(flat) == 0:
        raise ValueError("❌ After sliding window chunking, dataset is EMPTY.")

    print("🛠️ Setting up Trainer...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,  # effective batch size = PER_DEVICE_BATCH * accum * n_gpus
        learning_rate=3e-5,
        num_train_epochs=1,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        torch_compile=True,
        logging_steps=1000,
        dataloader_num_workers=min(6, max(1, os.cpu_count() // 2)),
        dataloader_pin_memory=True,
        save_steps=5000,
        save_total_limit=2,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="none",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=flat,
        data_collator=data_collator,
    )

    clean_gpu()

    checkpoint = None

    # if os.path.exists(OUTPUT_DIR):
    #     checkpoints = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
    #     if checkpoints:
    #         checkpoint = max(checkpoints, key=os.path.getctime)
    #         print(f"🔄 Resuming from checkpoint: {checkpoint}")
    #     else:
    #         print("📍 No checkpoint found, starting from scratch")

    if os.path.exists(OUTPUT_DIR):
        checkpoints = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
        
        if checkpoints:
            # Sort by checkpoint number
            checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]), reverse=True)
            
            for ckpt in checkpoints_sorted:
                weights_file = os.path.join(ckpt, "model.safetensors")
                if os.path.exists(weights_file):
                    file_size = os.path.getsize(weights_file) / (1024**2)  # MB
                    print(f"🔍 Checking {ckpt}: {file_size:.1f}MB")
                    
                    if file_size < 100:  # Less than 100MB is definitely corrupted
                        print(f"⚠️ Skipping corrupted checkpoint (too small): {ckpt}")
                        continue
                        
                    checkpoint = ckpt
                    print(f"✅ Valid checkpoint found: {checkpoint}")
                    break
            
            if checkpoint is None:
                print("❌ No valid checkpoints found, starting from scratch")
    else:
        print("📍 No checkpoint found, starting from scratch")

    print("🚀 Starting DAPT Pretraining...")
    # trainer.train()
    trainer.train(resume_from_checkpoint=checkpoint)

    print("💾 Saving model and tokenizer to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    clean_gpu()


if __name__ == "__main__":
    main()
