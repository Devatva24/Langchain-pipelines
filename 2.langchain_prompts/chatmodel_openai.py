from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature = 1.5, max_completion_tokens=10) 
# here if the temperature is low then the output generated will be similar to the previous generated one
# the difference betweeen consecutive outputs will differ as the value of the temperature is increased
result = model.invoke("What is the capital of India?")
print(result.content)