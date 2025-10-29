from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="The sentiment of the sent feedback either positive or negative")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative.\n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate response for the positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response for the negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : "The product is ugly"})
print(result)

# branch_chain = RunnableBranch(
#     (condition, chain),
#     (condition, chain),
#     (default, chain)
# )

# feedback = " The product quality is excellent. I loved it. Will buy again!"

# result = classifier_chain.invoke({'feedback': feedback})

# print(result.sentiment)

# template = prompt1.invoke({'feedback' : feedback})
# print(template)
# result = model.invoke(template)
# print(result.content)
