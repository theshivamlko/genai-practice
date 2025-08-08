from fastapi import  FastAPI
from ollama import Client
from fastapi.responses import JSONResponse

apiClient=FastAPI()

try:
    ollamaClient=Client(
        host="http://localhost:11434"
    )
   # ollamaClient.pull("gemma3:1b")
    print("Ollama initialized successfully")

except Exception as e:
    print('Error in connecting to ollama server:',e)
    exit(1)




@apiClient.get("/chat")
def chat(message: str):
    try:
        print('chat message ',message)
        response=ollamaClient.chat(model="gpt-oss:20b",messages=[
            {"role":"user","content":message}
        ])

        # Convert to serializable dict
        if hasattr(response, "model_dump"):
            return JSONResponse(content=response.model_dump())
        elif hasattr(response, "dict"):
            return JSONResponse(content=response.dict())
        else:
            return JSONResponse(content=response.__dict__)

    except Exception as e:
        print('Error in chat',e)
        return str(e)

print("Ollama API is running on http://localhost:8000/")




# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")
#
#
# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN"],
# )
#
# completion = client.chat.completions.create(
#     model="openai/gpt-oss-120b:cerebras",
#     messages=[
#         {
#             "role": "user",
#             "content": "Response hugging face link only. tell repo link of current model in use "
#         ,
#         }
#     ],
# )
#
# print(completion.choices[0].message)

