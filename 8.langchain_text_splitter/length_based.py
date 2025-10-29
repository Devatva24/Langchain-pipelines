from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate a report on the topic\n {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

chain = prompt | model | parser

result = chain.invoke({'topic' : 'AI'})

final_result = splitter.split_text(result)

print(final_result)

# result = splitter.split_text(text)