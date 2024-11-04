from llama_cpp import Llama
from tqdm import tqdm

llm = Llama.from_pretrained(
    repo_id="norallm/normistral-7b-warm",  # HuggingFace repository containing the GGUF files.
    filename="*Q4_K_M.gguf", # suffix of the filename containing the level of quantization. 
    n_ctx=768,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=4,            # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=0,# The number of layers to offload to GPU, if you have GPU acceleration available
    logits_all=True
)


#llm = Llama(
#      model_path="/itf-fi-ml/shared/ml-models/NoraLLm/normistral-7b-warm.Q8_0.gguf", 
#      n_threads=8,
#      n_gpu_layers=-1, # Uncomment to use GPU acceleration
#      # seed=1337, # Uncomment to set a specific seed
#      n_ctx=2048, # Uncomment to increase the context window
#)


f = open("pg13041.txt")
text = f.read()
f.close()

llm.reset()
text = text.encode()
tokens = llm.tokenize(text)
# les inn første token
llm.eval([tokens[0]])
for tk in tqdm(tokens[1:]):
    probs = llm.logits_to_logprobs(llm.eval_logits[-1])
    print(f"{llm.detokenize([tk]).decode()}\t{-probs[tk]}")
    llm.eval([tk])
