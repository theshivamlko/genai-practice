import os

from chatbot import create_chat_graph
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
MONGO_DB_URI = "mongodb://admin:admin@localhost:27017"

config = {"configurable": {"thread_id": "1"}}


# checkpointing, save I/O of every node
def init():
    with MongoDBSaver.from_conn_string(MONGO_DB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)

        while True:
            user_input = input("> ")

            for event in graph_with_mongo.stream({"messages": [{"role": "user", "content": user_input}]}, config,stream="value"):
                print(event)
                if "messages" in event["chatbot"]:
                    event["chatbot"]["messages"][-1].pretty_print()


init()
