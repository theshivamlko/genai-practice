import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")

modelName = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(modelName)

inputPrompt = [
    "The capital of India is"
]

tokenized = tokenizer(inputPrompt, return_tensors="pt")
inputIds = tokenized["input_ids"]
print("inputIds", inputIds)

autoModel = AutoModelForCausalLM.from_pretrained(modelName, torch_dtype=torch.bfloat16)

genResult = autoModel.generate(inputIds, max_new_tokens=25)

print("genResult", genResult)

decoded=tokenizer.batch_decode(genResult)

print("decoded", decoded)
