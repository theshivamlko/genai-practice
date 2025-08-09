import os
from typing import Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel




load_dotenv(os.path.join(os.path.dirname(__file__), '..','..', '.env'))

apiKey = os.getenv("OPENAI_API_KEY")

openClient = OpenAI(api_key=apiKey)


class DetectCallResponse(BaseModel):
    is_coding_questions: bool

class CodingCallResponse(BaseModel):
    answer: str


class StateChat(TypedDict):
    user_message: str
    ai_message: str
    is_coding_questions: bool


def detect_query(state: StateChat):
    user_message = state.get("user_message")

    result = openClient.beta.chat.completions.parse(

        model="gpt-4o",
        response_format=DetectCallResponse,
        messages=[
            {
                "role": "system",
                "content": "You are a coding expert. If user asks a coding question, "
                           "you will answer in True/ False format only "
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    print(result.choices[0].message.parsed.is_coding_questions)

    state["is_coding_questions"] = result.choices[0].message.parsed.is_coding_questions
    return state


def solve_coding_question(state: StateChat):
    user_message = state.get("user_message")

    result = openClient.beta.chat.completions.parse(

        model="gpt-4o",
        response_format=CodingCallResponse,
        messages=[
            {
                "role": "system",
                "content": "You are a coding expert. If user asks a coding question, "
                           "you will answer"
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )


    print("solve_simple_question===",result.choices[0].message.parsed.answer)

    state["ai_message"] = result.choices[0].message.parsed.answer


    return state


def solve_simple_question(state: StateChat):
    user_message = state.get("user_message")

    state["ai_message"] = "Please ask a coding question"

    return state


def router_edge(state: StateChat) -> Literal["SolveCodingQuestion", "SolveSimpleQuestion"]:
    if state["is_coding_questions"]:
        return "SolveCodingQuestion"
    else:
        return "SolveSimpleQuestion"


graph_builder = StateGraph(StateChat)

graph_builder.add_node("DetectQuery", detect_query)
graph_builder.add_node("SolveCodingQuestion", solve_coding_question)
graph_builder.add_node("SolveSimpleQuestion", solve_simple_question)
graph_builder.add_node("RouteEdge", solve_simple_question)

graph_builder.add_edge(START, "DetectQuery")
graph_builder.add_conditional_edges("DetectQuery", router_edge)

graph_builder.add_edge("SolveCodingQuestion", END)
graph_builder.add_edge("SolveSimpleQuestion", END)

graph = graph_builder.compile()


def call_invoke_graph():
    state = {
        "user_message": "if else",
        "ai_message": "",
        "is_coding_questions": False
    }

    result = graph.invoke(state)
    print("Final Result", result)


call_invoke_graph()
