from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Pick a Hugging Face embedding model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Create FAISS vector store from texts
vector_store = FAISS.from_texts(documents, embedding_function)

# Save FAISS index locally
vector_store.save_local("my_faiss_index")

# Load FAISS index back (e.g., in another script)
new_vector_store = FAISS.load_local("my_faiss_index", embedding_function, allow_dangerous_deserialization=True)

# Query
query = "tell me about bumrah"
results = new_vector_store.similarity_search(query, k=1)

print("Query:", query)
print("Best Match:", results[0].page_content)
