# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# documents = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# query = 'tell me about bumrah'

# doc_embeddings = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# scores = cosine_similarity([query_embedding], doc_embeddings)[0] # here in the output a 2D vector is given we then convert it into 1D by [0]

# index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
# enumerate puts an index to each score as per the doc list then it is sorted based on the score and
# [-1] takes the last from the sorted list which has the highest score

# print(query)
# print(documents[index]) # fetches the document with the highest score stored in the index
# print("similarity score is:", score)

from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Use a SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

# Generate embeddings
doc_embeddings = embedding_model.encode(documents)
query_embedding = embedding_model.encode([query])

# Similarity scores
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Best match
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query:", query)
print("Best Match:", documents[index])
print("Similarity Score:", score)

# Now ask Groq for a nice natural-language answer
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a cricket expert."},
        {"role": "user", "content": f"Answer the query: {query}\nContext: {documents[index]}"}
    ]
)

print("\nGroq Answer:", response.choices[0].message.content)

