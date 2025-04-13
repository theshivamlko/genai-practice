from google import genai
from google.genai.types import Content, GenerateContentConfig
from google.genai.types import Part
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=apiKey)

prompts = [
    Content(role='user', parts=[Part.from_text(text="List 10 best anime names till March 2025")]),
    # Content(role='system', parts=[Part.from_text(text="Be an Anime expert only")]),
]

response = client.models.generate_content(model='gemini-2.0-flash',
                                          contents=prompts, config=GenerateContentConfig(
        system_instruction=Content(role="system", parts=[
            Part.from_text(
            text="Be an Anime expert only. Don't answer on other topics. If asked on other topics Reply: I Cannot answer off-topic questions"),
            Part.from_text(
            text="Use this format to answer , "
                 "[Anime Name] - [YEAR] , Author: [Author Name], Genre: [Genre],"
                 "Main Characters: [Max 2 characters] ,"
                 "\nDescription: [Description]\n"),
        ]),
        temperature=1,
        # max_output_tokens=3,
    ))

print(f"Response => {response.text}")
