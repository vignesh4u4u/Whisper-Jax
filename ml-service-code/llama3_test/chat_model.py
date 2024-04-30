from transformers import pipeline,set_seed
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import torch
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","model","NousResearch/Meta-Llama-3-8B-Instruct"))
base_model_name = os.path.basename(path)
model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
def text_generation(query):
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = AutoConfig.from_pretrained(path)
    pipe = pipeline("text-generation",
                    model=path,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    config=config,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50, top_p=0.95
                    )
    messages = [
        {"role": "user", "content": query},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt,max_new_tokens=500)
    final_answer = outputs[0]["generated_text"]
    return ({"result":final_answer})
