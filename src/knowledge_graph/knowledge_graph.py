import os

import dotenv

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY, "model": 'text-embedding-3-small'
        },
        "llm": {
            "provider": "openai", "config": {
                "api_key": OPENAI_API_KEY,
                "model": "gpt-4.1"
            }
        },
        "vector_store":{
            "provider":"neo4j",
            "config":{
                "url":NEO4J_URL,
                "username":NEO4J_USERNAME,
                "password":NEO4J_PASSWORD
            }
        }
    }
}
