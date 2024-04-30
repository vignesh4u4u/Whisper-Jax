import ollama
response = ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': 'please explaint the concept of deep learning'}])
assistant_response = response['message']['content']
print(assistant_response)
