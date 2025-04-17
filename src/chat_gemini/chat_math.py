from google import genai
from google.genai.types import Content, GenerateContentConfig
from google.genai.types import Part
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=apiKey)

prompts = [
    Content(role='user', parts=[Part.from_text(text="Whats is greater 9.8 or 9.11 ?")]),
]

response = client.models.generate_content(model='gemini-2.0-flash',
                                          contents=prompts, config=GenerateContentConfig(
        # chain of thoughts prompting
        system_instruction=Content(role="system", parts=[
            Part.from_text(
                text="""Follow strict JSON for output instruction:"
                 "
                 Example:
                 Input: Whats 2+2 ?
                 Output: {{"step":"analyze","content":"Alright! Its a Algebra Question Question"}}
                 Output: {{"step":"output","content":"The answer is 4"}}
                 Output: {{"step":"validate","content":"The question is addition so The answer is define 4 and correct"}}
                 Output: {{"step":"action","content":"addFunction"}}
                 
                 Example:
                 Input: Whats sqrt of 144 ?
                 Output: {{"step":"analyze","content":"Alright! Its a Algebra Question Question"}}
                 Output: {{"step":"output","content":"The answer is 12"}}
                 Output: {{"step":"validate","content":"The question is Sqaure root so The answer is define 12 and correct"}}
                 Output: {{"step":"action","content":"sqrtFunction"}}
                 
                 """),

        ]),
        temperature=0,
        # max_output_tokens=3,
    ))

print(f"Response => {response.text}")
