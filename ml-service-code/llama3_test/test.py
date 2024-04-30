from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","Meta-Llama-3-8B-Instruct","Meta-Llama-3-8B-Instruct.Q2_K.gguf"))

generator = LlamaCppGenerator(
    model=path,
    n_ctx=512,
    #n_batch=128,
    model_kwargs={"n_gpu_layers": -1,"seed":42},
		generation_kwargs={"max_tokens": 400, "temperature": 0.5},
)
generator.warm_up()
prompt = "what is the use of copper?"
result = generator.run(prompt)

output = result
filtered_output = [{'text': output['replies'][0], 'index': 0, 'logprobs': None, 'finish_reason': 'length'}]
final_answer = filtered_output[0]['text']
print(final_answer)