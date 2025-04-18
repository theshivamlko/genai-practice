import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")

modelName = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(modelName)

result=tokenizer("Hello world!")

print("Tokenizer of Gemma Model", )
print("Is Cuda available", torch.cuda.is_available())
# print tokenizer.get_vocab())

input_ids=result["input_ids"]
print(input_ids)

autoModel=AutoModelForCausalLM.from_pretrained(modelName,torch_dtype=torch.bfloat16)

genPipeline=pipeline("text-generation",model=modelName,tokenizer=tokenizer)

response=genPipeline("Hey There",max_new_tokens=25)

print(response)


