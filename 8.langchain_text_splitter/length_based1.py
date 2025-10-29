from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0, # helps to prevent losing the context; common portion between two consecutive chunks = 10 to 20% is preferrable
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)