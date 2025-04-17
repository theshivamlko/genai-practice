
from google import genai
from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, Part, GenerateContentResponse
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")

gemini: Client = genai.Client(api_key=apiKey)

promptlist: list[Content] = [
    Content(role="user", parts=[
        Part.from_text(text="WHats the current weather of Lucknow")
    ])
]

systemPrompt = Content(
    role="system",
    parts=[
        Part.from_text(text="You are an helpfull assistant who is specialized in resolving user query."
                            "You work on start,plan,action observer mode."
                            "For the given query follow the step by step execution."
                            "Select the relevant tool from the available tool."
                            "Wait for the observation and based on observation from tool , resolve the user query")

    ]
)

response: GenerateContentResponse = gemini.models.generate_content(
    model="gemini-2.0-flash",
    contents=promptlist,
    config=GenerateContentConfig(
        system_instruction=systemPrompt

    )
)

print(response.text)
