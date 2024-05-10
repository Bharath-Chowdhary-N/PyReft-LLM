import torch, transformers, pyreft, load_dotenv, os
from dotenv import load_dotenv
import pandas as pd
from huggingface_hub import login
login(token=os.getenv('HF_TOKEN'))

#load dotenv 
load_dotenv()

#add and load model from hugging face
#model_name = 'nvidia/Llama3-ChatQA-1.5-8B'
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda:0', cache_dir='./cache', token=os.getenv('HF_TOKEN'))
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

def prompt_template(prompt):
    response = f"""<s>[INST]<<sys>> you are a helpful assistant <</sys>> 
                   {prompt}
                   [/INST]"""
    return response

generation_params = {
    'max_length': 200,  # Set max_length to accommodate the length of your input_ids
    # or alternatively,
    #'max_new_tokens': 5,  # Set max_new_tokens to the additional number of tokens you want to generate
}

prompt = prompt_template("Does BCN have a vision? if yes what is his vision")
tokens = tokenizer(prompt, return_tensors='pt').to('cuda')

reft_model = pyreft.ReftModel.load('./model_temp_save',model)
reft_model.set_device('cuda')


base_unit_position = tokens['input_ids'].shape[-1]-1
_, response = reft_model.generate(tokens, unit_locations={'sources->base':(None,[[[base_unit_position]]])}
           , intervene_on_prompt=True
)


print(tokenizer.decode(response[0]))
#rint(tokenizer.decode(tokens))
