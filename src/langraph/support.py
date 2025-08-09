import json
import os


from chatbot import create_chat_graph
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
MONGO_DB_URI = "mongodb://admin:admin@localhost:27017"


# resume chat with the same thread
# if the thread_id is not set, a new thread will be created
config = {"configurable": {"thread_id": "8"}}


# checkpointing, save I/O of every node
def init():
    with MongoDBSaver.from_conn_string(MONGO_DB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        state=graph_with_mongo.get_state(config=config)
        for message in state.values["messages"]:
            print(message.pretty_print())

        last_message= state.values["messages"][-1]
        tools_calls = last_message.additional_kwargs.get("tool_calls", [])
        user_query= None

        if tools_calls:
            print("Tool calls:")
            for tool_call in tools_calls:
                name=tool_call.get("function",{}).get("name", "unknown_tool")
                arguments=tool_call.get("function",{}).get("arguments", "{}")

                print(f"Tool: {name}, Arguments: {arguments}")
                if name == "human_assisyance_tool":
                    args=json.loads(arguments)
                    user_query=args.get("query", None)

            print("Human assistance requested for Query: =====", user_query)
            answer =input("Resolution >")

            resume=Command(resume={"data": answer})

            for event in graph_with_mongo.stream(resume,config,stream_mode="values"):
                print(event)
                if "messages" in event:
                    event["messages"][-1].pretty_print()




init()
