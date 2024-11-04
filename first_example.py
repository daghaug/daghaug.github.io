from llama_cpp import Llama

llm = Llama(
      model_path="/itf-fi-ml/shared/ml-models/NoraLLm/normistral-7b-warm.Q8_0.gguf", 
      n_threads=8,
      n_gpu_layers=-1,
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

