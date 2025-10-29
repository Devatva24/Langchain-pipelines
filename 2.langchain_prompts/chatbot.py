# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

chat_history = [SystemMessage(content='You are a helpful assistant')]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content = user_input))
    if(user_input == 'exit'):
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("Bot: ", result.content)

print(chat_history)