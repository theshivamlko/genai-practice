# import os
# import torch
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#
# load_dotenv()
#
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#
# modelName = "openai/gpt-oss-20b"
#
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()
#
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(modelName)
#
# # BitsAndBytesConfig for 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )
#
# # Load the quantized model
# model = AutoModelForCausalLM.from_pretrained(
#     modelName,
#     device_map="auto",
#     trust_remote_code=True,
#     quantization_config=bnb_config,
# )
#
# # Prepare inputs
# input_text = "Hey there!"
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
#
# # Generate output
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         do_sample=True,
#         top_p=0.9,
#         temperature=0.8
#     )
#
# # Decode and print
# response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response_text)


from transformers import pipeline
import torch

model_id = "openai/gpt-oss-120b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])