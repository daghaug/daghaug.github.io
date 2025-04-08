from llama_cpp import Llama

# hvis du bruker ferdignedlastede modeller på ml-nodene bruker du denne linja
# llm = Llama(
#    model_path="/itf-fi-ml/shared/ml-models/NoraLLm/normistral-7b-warm.Q8_0.gguf",
#    n_threads=8,
#    n_gpu_layers=32,
#    # seed=1337, # Uncomment to set a specific seed
#    n_ctx=2048, # Set context window size
#)


# Og hvis du skal laste ned modellen fra HugginFace bruker du denne
llm = Llama.from_pretrained(
    repo_id="norallm/normistral-7b-warm",  # HuggingFace repository containing the GGUF files.
    filename="*Q8_0.gguf", # suffix of the filename containing the level of quantization. 
    n_threads=8,
    n_gpu_layers=32,
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=2048, # Set context window size
)


# Simple inference example
output = llm(
  "Engelsk: Hello everyone! I'm a language model, how are you doing today?\nBokmål:", # Prompt
  max_tokens=128,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token
  echo=True,       # Whether to echo the prompt
    seed=-1,
  temperature=0.3  # Temperature to set, for Q3_K_M, Q4_K_M, Q5_K_M, and Q6_0 it is recommended to set it relatively low.
)
print(output)

