import langchain
print(langchain.__version__)

import os
print("Hugging Face Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

from dotenv import load_dotenv, find_dotenv
import os

# Find and load the .env file
env_path = find_dotenv()
print("Found .env at:", env_path)
load_dotenv(dotenv_path=env_path)

# Check if token is loaded
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Loaded token:", token)
