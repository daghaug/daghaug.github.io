from llama_cpp import Llama

llm = Llama(
      model_path="/itf-fi-ml/shared/ml-models/NoraLLm/normistral-7b-warm.Q8_0.gguf", 
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      n_ctx=2048, # Uncomment to increase the context window
)

# Simple inference example


yrker = ["advokaten", "legen", "feieren", "ungkaren", "sykepleieren", "mordersken", "morderen"]

for yrke in yrker:
    output = llm(
        f"Den tiltalte {yrke} påstod at dommeren hadde et bein i siden til",   
        seed=-1,
        max_tokens=1, 
        stop=[".", "</s>", "\n"],   # Example stop token
        echo=True,       # Whether to echo the prompt
        temperature=20000,
        #        top_k=200,
    )

    print(output["choices"][0]["text"])

