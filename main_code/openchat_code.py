from transformers import pipeline
import torch
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","model","openchat-3.5-0106"))
model_name = os.path.basename(path)
pipe = pipeline("text-generation", model=path, torch_dtype=torch.bfloat16, device_map="auto")
def text_generation(query):
    messages = [
        {"role": "user", "content": query},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    final_answer = outputs[0]["generated_text"]
    return ({"result":final_answer})
