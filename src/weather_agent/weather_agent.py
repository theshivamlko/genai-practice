import json

import requests
from google import genai
from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, Part, GenerateContentResponse
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("GEMINI_API_KEY")

gemini: Client = genai.Client(api_key=apiKey)

# userQuery =input('> ')
userQuery = "Whats weather in  Hyderabad"


def get_weather(location: str) -> str:
    url=f"""https://wttr.in/{location}?format=%C+%t"""

    response=requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return "Error: Unable to fetch weather data."


availableTools = {
    "get_weather": {
        "func": get_weather,
        "description": "Get the weather of a location from Weather API",

    }
}

availableToolsStr = ""
for tool in availableTools.keys():
    availableToolsStr += f"{tool} : {availableTools[tool]['description']}\n"

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
        {"step": "action", "content": "getWeather", "function": "getWeather", "input_params": "New York"},
    ]
}
system_prompt_text = f"""You are a helpful assistant who operates in a 'start, plan, action, observer' mode to resolve user queries step by step.
The output MUST be a single, valid JSON object. Do not include any markdown formatting, code blocks, or extraneous text around the JSON.
The JSON object will contain an array named 'steps', where each element represents a step.
Each step has 'step', 'content', and optionally 'function' and 'input_params' for 'action' steps.
Follow the 'start, plan, action, observer' cycle meticulously.

Rules:
1. The final output MUST be a single.
2. Each element in the 'steps' array follows the specified JSON format: {json.dumps(output_format['steps'][0])}
4. Perform 1 step at a time and wait for its result.
5. Pass result from a step to next step. represented as a JSON object within as 'step'.
6. If a step fails, include an 'error' step with a detailed error content and then conclude.
7. If no error found , then strictly complete all steps.

Output format: {json.dumps(output_format)}

Available tools:
{availableToolsStr}

 
Example:
User query: What's the weather like in Tokyo?
Output: {json.dumps(example_output)}
"""


# print(system_prompt_text)

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

for step in jsonData["steps"]:

    if step["step"] == "error":
        print("5. Error in processing the request: => ", step['content'])

    if step["step"] == "plan":
        print("1. Plan: => ", step['content'])

    if step["step"] == "action":
        print("2. Action Performed: => ", step['content'])
        toolName = step['function']
        toolParams = step['input_params']

        if availableTools.get(toolName, False):
            output = availableTools[toolName].get('func')(toolParams)
            print("     Action Result: => ", output)


    if step["step"] == "observer":
        print("3. Observation: => ", step['content'])

    if step["step"] == "output":
        print("4. Output Performed: => ", step['content'])
