import os

def HugggingfaceModel():
    text_prompt = "<s>[INST] What is your favourite condiment? [/INST]"
    "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
    "[INST] Do you have mayonnaise recipes? [/INST]"
    model_name = "HUGGINGFACEHUB_API_TOKEN"
    return model_name
def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt
def HuggingfaceAccess():
    model_valied_format1 = "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"
    return model_valied_format1
def HuggingfaceAccess1():
    model_valied_format1 = "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"
    return model_valied_format1
def Huggingface_mistral():
    model_valied_format1 = "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"
    return model_valied_format1
def groq_model_name():
    model_name_load = "GROQ_API_KEY"
    return model_name_load
def groq_model_accesess():
    model_required = "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"
    return model_required
