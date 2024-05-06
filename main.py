import torch, transformers, pyreft, load_dotenv, os
from dotenv import load_dotenv

#load dotenv 
load_dotenv()

#add and load model from hugging face
model_name = 'meta-llama/Llama-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='cuda:0', cache_dir='./cache', token=os.getenv('HF_TOKEN'))
#Reason: torch_dtype=torch.float16 is needed to train with Cuda
#        cache_dir is the place where the output is stored when used in cloud
#        get hugging face token

# Add tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_tokens = 2048, use_fast = False, padding_side='right', token=os.getenv('HF_TOKEN'))
tokenizer.pad_token = tokenizer.unk_token
#Reason: Slow tokenizers (use_fast=False) are those written in Python inside the Transformers library, while the fast versions (use_fast=True) are the ones provided by Tokenizers, which are written in Rust.
#        model_max_tokens is the maximum number of tokens that can be processed by the model.
#        padding_side is the side on which the model will be applied.
#        get hugging face token

