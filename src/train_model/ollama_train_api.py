from fastapi import  FastAPI
from ollama import Client


apiClient=FastAPI()

try:
    ollamaClient=Client(
        host="http://localhost:11434"
    )
    ollamaClient.pull("gemma3:1b")

except Exception as e:
    print('Error in connecting to ollama server:',e)
    exit(1)




@apiClient.get("/chat")
def chat(message: str):
    try:
        print('chat message ',message)
        response=ollamaClient.chat(model="gemma3:1b",messages=[
            {"role":"user","content":message}
        ])

        return  response.get("message").get("content")
    except Exception as e:
        print('Error in chat',e)
        return str(e)
