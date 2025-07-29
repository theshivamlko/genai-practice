import os

import dotenv

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_HOST = os.getenv("QDRANT_HOST")

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-large"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "gpt-4o"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "collection_name": "knowledge_graph",
        "config": {
            "host": "localhost",
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        }
    }
}

mem_client = Memory.from_config(config)

openaiClient = OpenAI(api_key=OPENAI_API_KEY)


def chat(message):
    try:
        mem_result = mem_client.search(query=message, user_id="p123")
    except Exception as e:
        print(f"Error during memory search: {e}")


    print(f"\n\nMEMORY RAW :\n\n{mem_result }\n\n")



    memories = "\n".join([m["memory"] for m in mem_result.get("results")])

    print(f"\n\nMEMORY:\n\n{memories}\n\n")

    SYSTEM_PROMPT = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling

        Memory and Score:
        {memories}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    result = openaiClient.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    messages.append(
        {"role": "assistant", "content": result.choices[0].message.content}
    )

    mem_client.add(messages, user_id="p123")
    return result.choices[0].message.content


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = chat(user_input)
    print(f"Assistant: {response}")
