from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class StateChat(TypedDict):
    user_message: str
    ai_message: str
    is_coding_questions: bool


def detect_query(state: StateChat):
    user_message = state.get("user_message")

    state["is_coding_questions"] = True
    return state


def solve_coding_question(state: StateChat):
    user_message = state.get("user_message")

    state["ai_message"] = "Here is you coding question  answer"


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
        "user_message": "How are you ?",
        "ai_message": "",
        "is_coding_questions": False
    }

    result = graph.invoke(state)
    print("Final Result", result)


call_invoke_graph()
