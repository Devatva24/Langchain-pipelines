from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

loader = TextLoader('cricket.txt', encoding='utf-8')

load_dotenv()

prompt = PromptTemplate(
    template='Write the summary of the poem {poem}',
    input_variable=['poem']
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))

# print(docs)

# print(docs[0])

# print(docs[0].page_content)

# print(docs[0].metadata)

# print(type(docs))