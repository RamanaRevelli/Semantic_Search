from flask import Flask, render_template, request
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)

chroma_client = chromadb.PersistentClient(path="my_vectordb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="msmarco-bert-base-dot-v5")
collection = chroma_client.get_collection(name="sub", embedding_function=sentence_transformer_ef)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = collection.query(
            query_texts=[query],
            n_results=10,
            include=['documents', 'distances', 'metadatas']
        )

        if results:
            result_data = []
            for j in range(min(len(results['ids']), len(results['distances']), len(results['documents']), len(results['metadatas']))):
                id_list = results["ids"][j]
                distance_list = results['distances'][j]
                document_list = results['documents'][j]
                metadata_list = results['metadatas'][j]

                for id, distance, document, metadata in zip(id_list, distance_list, document_list, metadata_list):
                    result_data.append({'id': id, 'distance': distance, 'document': document, 'metadata': metadata})

            return render_template('results.html', results=result_data)
        else:
            return render_template('results.html', message="No results found.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
