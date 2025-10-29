from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about the {topic}',
    input_variable=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the given joke :  {joke}',
    input_variable=['joke']
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_generator = RunnableSequence(prompt1, model, parser)

def word_counter(text):
    return len(text.split())

passthrough = RunnablePassthrough()

parallel_chain = RunnableParallel({
    'passthrough':passthrough,
    'meaning':RunnableSequence(prompt2, model, parser),
    'word_count':RunnableLambda(word_counter)
})

join_chain = RunnableSequence(joke_generator, parallel_chain)

result = join_chain.invoke({'topic' : 'Space'})
print(result)