import torch, transformers, pyreft

#add and load model from hugging face
model_name = 'meta-llama/Llama-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='cuda:0', cache_dir='./cache', token='<KEY>')