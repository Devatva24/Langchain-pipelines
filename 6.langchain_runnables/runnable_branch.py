from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on this topic : {topic}',
    input_variable=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the report : {report}',
    input_variable=['report']
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

report_generation = RunnableSequence(prompt1, model, parser)

def word_count_long(text) -> bool:
    return len(text.split()) > 500

def word_count_short(text) -> bool:
    return len(text.split()) <= 500

passthrough = RunnablePassthrough()

runnable_branch = RunnableBranch(
    (word_count_long , RunnableSequence(prompt2, model, parser)),
    (word_count_short , passthrough),
    RunnableLambda(lambda x:"None of the output matches")
)

join_chain = RunnableSequence(report_generation, runnable_branch)

result = join_chain.invoke({'topic':'Black hole'})

print(result)