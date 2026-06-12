from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from vllm.distributed.parallel_state import destroy_model_parallel
import subprocess
import torch
import os
import gc

def check_gpu():
   result = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
      stdout=subprocess.PIPE, text=True
   )
   print("GPU Memory:", result.stdout.strip())

def check_cpu():
   total, used, free = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
   print("RAM: ", used, " (used)", free, " (free)")

prompts = [
   "Fix this grammar: I are going to the store.",
   "Correct this sentence: She don't like apples.",
   "Sing me a song.",
   "Translate this to Spanish: How are you today?",
   "Summarize this: Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence.",
   "Explain this like I'm five: Why is the sky blue?",
   "Make this sentence more formal: Gimme that report ASAP.",
   "Turn this into a haiku: The sun sets slowly / Painting the clouds with bright fire / Night embraces all.",
   "Fix this grammar: I are going to the store.",
   "Correct this sentence: She don't like apples.",
   "Sing me a song.",
   "Translate this to Spanish: How are you today?",
   "Summarize this: Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence.",
   "Explain this like I'm five: Why is the sky blue?",
   "Make this sentence more formal: Gimme that report ASAP.",
   "Turn this into a haiku: The sun sets slowly / Painting the clouds with bright fire / Night embraces all.",
]

sampling_params = SamplingParams(
                  n=1,
                  temperature=0.0,
                  top_p=1.0, # disable nucleus filtering
                  top_k=0, # disable top-k filtering (greedy decode)
                  seed=42,
                  min_tokens=1,
                  repetition_penalty=1.0
                  )

llm = LLM(model="mistralai/Mistral-Nemo-Instruct-2407", # TODO: Try with AWQ model
      tokenizer_mode="mistral",
      load_format="mistral",
      config_format="mistral",
      max_model_len=8192, # TODO: Increase
      gpu_memory_utilization=0.875,
      seed=42,
      swap_space=64,
      cpu_offload_gb=100,
      dtype="float16", # To speed stuff
      # max_seq_len_to_capture=4096,
      tensor_parallel_size=2,
      # max_num_seqs=2048,
      enable_prefix_caching=True,
      # enable_chunked_prefill=True,
      task='generate',
      disable_log_stats=True,
      enforce_eager=False,
      block_size=32,
      )

### THIS WORKS
# llm = LLM(model="GameScribes/Mistral-Nemo-AWQ",
#       max_model_len=2048,
#       gpu_memory_utilization=0.85,
#       seed=42,
#       swap_space=30,
#       cpu_offload_gb=20,
#       dtype=torch.bfloat16,
#       quantization="AWQ")

check_gpu()
check_cpu()

outputs = llm.generate(prompts, sampling_params) # Trying with batch of 16 prompts

check_gpu()
check_cpu()

for output in outputs:
   print(f"Prompt: {output.prompt}")
   print(f"Generated: {output.outputs[0].text}")
   print("-" * 50)

destroy_model_parallel()
del llm.llm_engine.model_executor.driver_worker
del llm # Isn't necessary for releasing memory, but why not
gc.collect()
torch.cuda.empty_cache()