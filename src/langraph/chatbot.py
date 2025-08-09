import os
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

apiKey = os.getenv("OPENAI_API_KEY")

# openClient = OpenAI(api_key=apiKey)

llm = init_chat_model(model_provider="openai", model="gpt-4o")


class DetectCallResponse(BaseModel):
    is_coding_questions: bool


class CodingCallResponse(BaseModel):
    answer: str


class StateChat(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: StateChat):
    messages = state.get("messages")
    response = llm.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(StateChat)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Without memory
graph = graph_builder.compile()

# create new graph with checkpointer
def create_chat_graph(checkpointer):
    return  graph_builder.compile(checkpointer=checkpointer)
