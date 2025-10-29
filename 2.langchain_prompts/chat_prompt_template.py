from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} assistant'),
    ('human', 'Tell me about {topic}'),
    # SystemMessage(content='You are a helpful {domain} assistant'),
    # HumanMessage(content='Tell me about {topic}'),
])

prompt = chat_template.invoke({'domain':'AI', 'topic':'LangChain'})
print(prompt)