import os
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.types import Command
from  langgraph.prebuilt import ToolNode, tools_condition



load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

apiKey = os.getenv("OPENAI_API_KEY")

@tool()
def human_assisyance_tool(query:str):
    """
    Request human assistance for a specific query.
    """
    human_response=interrupt({"query": query}) # graph will exit and wait for human response

    return human_response["data"] # This will be used to resume the graph later

tools=[human_assisyance_tool]
tool_node=ToolNode(tools=tools)

llm = init_chat_model(model_provider="openai",model="gpt-4o")
llm__with_tools =llm.bind_tools(tools=tools)



class DetectCallResponse(BaseModel):
    is_coding_questions: bool


class CodingCallResponse(BaseModel):
    answer: str


class StateChat(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: StateChat):
    messages = state.get("messages")
    response = llm__with_tools.invoke(messages)
    assert  len(response.tool_calls) <=1
    return {"messages": [response]}


graph_builder = StateGraph(StateChat)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")


graph_builder.add_conditional_edges( "chatbot",tools_condition)
graph_builder.add_edge("chatbot", END)

# Without memory
graph = graph_builder.compile()

# create new graph with checkpointer
def create_chat_graph(checkpointer):
    return  graph_builder.compile(checkpointer=checkpointer)
