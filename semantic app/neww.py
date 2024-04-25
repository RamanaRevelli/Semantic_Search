import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Check if the collection already exists
if "sub" not in chroma_client.list_collections():
    # Collection does not exist, create it
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="msmarco-bert-base-dot-v5")
    collection = chroma_client.create_collection(name="sub", embedding_function=sentence_transformer_ef)
else:
    # Collection already exists
    collection = chroma_client.get_collection(name="sub")

# Now you can proceed with querying the collection
