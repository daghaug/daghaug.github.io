from llama_cpp import Llama


llm = Llama.from_pretrained(
  repo_id="norallm/normistral-7b-warm-instruct",  # HuggingFace repository containing the GGUF files.
  filename="*Q4_K_M.gguf", # suffix of the filename containing the level of quantization. 
  n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
)

# Vi prøver et system prompt, men det funker ikke særlig bra. Kanskje fordi modellstørrelsen er liten? Prøve dette på ml9 også
system_prompt = {"role":"user",
                 "content":"Uansett hva jeg spør om, skal du svare i tung, ugjennomtrengelig byråkratstil. Hvert svar skal være bare én setning."}


messages = [system_prompt]
prompt = input("User: ")

while prompt != "STOPP":
    messages.append({"role":"user",
                       "content":prompt})
    output = llm.create_chat_completion(messages=messages, temperature=1.0)
    answer = output["choices"][0]["message"]["content"]
    print("Assistent: ", answer)
    messages.append({"role":"assistant",
                        "content":answer})
    prompt = input("User: ")
