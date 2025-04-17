import json

from google import genai
from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, Part, GenerateContentResponse
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")

gemini: Client = genai.Client(api_key=apiKey)

# userQuery =input('> ')
userQuery = "Whats weather in  Lucknow"


def getWeather(location: str) -> str:
    return "30 degree celsius"


promptlist: list[Content] = [
    Content(role="user", parts=[
        Part.from_text(text=userQuery)
    ])
]

output_format = {
    "steps": [
        {
            "step": "string",
            "content": "string",
            "function": "string (optional)",
            "input_params": "string (optional)"
        }
        # ... more steps will be added here
    ]
}

# Define the example output as a Python dictionary
example_output = {
    "steps": [
        {"step": "plan", "content": "User wants to know the weather in New York."},
        {"step": "plan", "content": "I will use the 'getWeather' tool to retrieve this information."},
        {"step": "action", "content": "getWeather", "function": "getWeather", "input_params": "New York"}
    ]
}

system_prompt_text = f"""You are a helpful assistant who operates in a 'start, plan, action, observer' mode to resolve user queries step by step.
The output MUST be a single, valid JSON object. Do not include any markdown formatting, code blocks, or extraneous text around the JSON.
The JSON object will contain an array named 'steps', where each element represents a step.
Each step has 'step', 'content', and optionally 'function' and 'input_params' for 'action' steps.
Follow the 'start, plan, action, observer' cycle meticulously.

Rules:
1. The final output MUST be a single, valid JSON object WITHOUT any additional formatting.
2. The JSON object must contain a 'steps' array.
3. Each element in the 'steps' array follows the specified JSON format: {json.dumps(output_format['steps'][0])}
4. Perform 1 step at a time, represented as a JSON object within the 'steps' array.
5. If a step fails, include an 'error' step with a detailed error message and then conclude.

Output format: {json.dumps(output_format)}

Example:
User query: What's the weather like in Tokyo?
Output: {json.dumps(example_output)}
"""

systemPrompt = Content(
    role="system",
    parts=[
        Part.from_text(text=system_prompt_text)

    ]
)

response: GenerateContentResponse = gemini.models.generate_content(
    model="gemini-2.0-flash",
    contents=promptlist,

    config=GenerateContentConfig(
        system_instruction=systemPrompt,
        response_mime_type='application/json',

    ),

)

print(response.text)


jsonData = json.loads(response.text)

# print(jsonData.action)
# print(jsonData.get("action"))
