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
userQuery = "Whats weather in Lucknow"


def get_weather(location: str) -> str:
    url = f"""https://wttr.in/{location}?format=%C+%t"""

    response = requests.get(url)

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

output_format = """
 {
            "step": "string",
            "content": "string",
            "function": "string (optional)",
            "input_params": "string (optional)"
        } """

# Define the example output as a Python dictionary
example_output = """

This step tell what to user is interested in and what is the next step to be performed.
      Output: {{ "step": "plan", "content": "The user is interested in weather data of new york" }}
      
      This step tell which tool to look for in available tools for the next step.
      Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
      
    This step tell tool name and input params to be used in given format.
     Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
     
    This step understand the output.
     Output: {{ "step": "observe", "output": "12 Degree Cel" }}
     
    This step tell the final output to user friendly format.
     Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
    """

system_prompt_text = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tools. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.
    
Rules:
1. The final output MUST be a single JSON Object.
2. Each output has step follows the specified JSON format: {json.dumps(output_format)}
3. Always perform one step at a time and wait for next input.
4. If a step fails, include an 'error' step with a detailed error content and then conclude.
5. Each step proceed to next step

Available tools:
{availableToolsStr}

 
Example:
User query: What's the weather like in Tokyo?

Output for each step
 {json.dumps(example_output)}
"""

# print(system_prompt_text)

systemPrompt = Content(
    role="system",
    parts=[
        Part.from_text(text=system_prompt_text)
    ]
)

while True:

    response: GenerateContentResponse = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=promptlist,
        config=GenerateContentConfig(
            system_instruction=systemPrompt,
            response_mime_type='application/json',
        ),
    )

    print(response.text, "\n\n")

    jsonData = json.loads(response.text)

    if jsonData["step"] == "error":
        print("5. Error in processing the request: => ", jsonData['content'])

    if jsonData["step"] == "plan":
        print("1. Plan: => ", jsonData['content'])
        #
        # promptlist = [
        #     Content(role="user", parts=[
        #         Part.from_text(text=json.dumps({
        #             "step": "plan",
        #             "content": jsonData['content']
        #         }))
        #     ])
        # ]
        continue

    if jsonData["step"] == "action":
        print("2. Action Performed: => ", jsonData['content'], jsonData['function'])
        toolName = jsonData['function']
        toolParams = jsonData['input_params']

        if availableTools.get(toolName, False):
            output = availableTools[toolName].get('func')(toolParams)
            print("     Action Result: => ", output)

            promptlist = [
                Content(role="user", parts=[
                    Part.from_text(text=json.dumps({
                        "step": "observer",
                        "output": jsonData['content']
                    }))
                ])
            ]
            continue

    if jsonData["step"] == "output":
        print("4. Output Performed: => ", jsonData['content'])

        promptlist = [
            Content(role="user", parts=[
                Part.from_text(text=jsonData['content'])
            ])
        ]
        break
