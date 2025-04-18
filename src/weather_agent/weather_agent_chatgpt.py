import json
from time import sleep
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
apiKey = os.getenv("CHATGPT_API_KEY")

client = OpenAI(api_key=apiKey)

userQuery = input('> ')


# userQuery = "Whats weather in Lucknow and Hyderabad in Fahrenheit ."

def get_weather(cityName: str) -> str:
    url = f"""https://wttr.in/{cityName}?format=%C+%t"""

    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return "Error: Unable to fetch weather data."


def query_db(sql: str):
    pass


def run_command(command: str):
    result = os.system(command)
    return result


availableTools = {
    "get_weather": {
        "func": get_weather,
        "description": "Get the weather of a city from Weather API",
    },
    "run_command": {
        "func": run_command,
        "description": "Run a script cli",
    }
}

availableToolsStr = ""
for tool in availableTools.keys():
    availableToolsStr += f"{tool} : {availableTools[tool]['description']}\n"

output_format = """{
            "step": "string",
            "content": "string",
            "function": "string (optional)",
            "input_params": "string (optional)"
        } """

example_output = """
This step tell what to user is interested in and what is the next step to be performed.
     Plan Output: {{ "step": "plan", "content": "The user is interested in weather data of New york" }}

      This step tell which tool to look for in available tools for the next step.
     Plan Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}

    This step tell tool name and input params to be used in given format.
    Action Output: {{ "step": "action", "function": "get_weather", "input": "New york" }}

    This step understand the output.
     Observe Output: {{ "step": "observe", "output": "12 Degree Cel" }}

    This step tell the final output to user friendly format.
     Output: {{ "step": "output", "content": "The weather for New york seems to be 12 degrees." }}
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


Sample Example:
User query: What's the weather in New York?

Output for each step
 {json.dumps(example_output)}

 User query: Whats weather in New York and Boston?

Output for each step
 Plan Output: {json.dumps({"step": "plan", "content": "The user is interested in weather data of New York and Boston"})}

 Plan Output: {json.dumps({"step": "plan", "content": "I will use the 'get_weather' tool to retrieve this information."})}

 Action Output: {json.dumps({"step": "action", "function": "get_weather", "input_params": "New York"})}

 Action Output: {json.dumps({"step": "action", "function": "get_weather", "input_params": "Boston"})}

 Observe Output: {json.dumps({"step": "observe", "output": "15°C and Cloudy in New York"})}

 Observe Output: {json.dumps({"step": "observe", "output": "18°C and Cloudy in Boston"})}

 Output: {json.dumps({"step": "output", "content": "The weather for New York is approximately 15°C and Boston is 18°C."})}


"""

messages = [
    {"role": "system", "content": system_prompt_text},
    {"role": "user", "content": userQuery}
]

while True:
    sleep(2)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    print("Response=>\n", response.choices[0].message.content, "\n\n")
    jsonData = json.loads(response.choices[0].message.content)

    if jsonData["step"] == "error":
        print("5. Error in processing the request: => ", jsonData['content'])
        break

    if jsonData["step"] == "plan":
        print("1. Plan: => ", jsonData['content'])

        messages.append({
            "role": "user",
            "content": json.dumps({
                "step": "plan",
                "content": jsonData['content']
            })
        })
        continue

    if jsonData["step"] == "action":
        print("2. Action Performed: => ", jsonData['function'], jsonData['input_params'])
        toolName = jsonData['function']
        toolParams = jsonData['input_params']

        if availableTools.get(toolName, False):
            output = availableTools[toolName].get('func')(toolParams)
            print("     Action Result: => ", output)

            messages.append({
                "role": "user",
                "content": json.dumps({
                    "step": "observe",
                    "output": f'Output from action steps is {output}'
                })
            })
            continue

    if jsonData["step"] == "output":
        print("4. Output Performed: => ", jsonData['content'])
        break
