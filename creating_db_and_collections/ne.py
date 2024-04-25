import chromadb
from chromadb.utils import embedding_functions
import time
import multiprocessing as mp
import csv
csv.field_size_limit(100000000)
def producer(filename, batch_size, queue):

    with open(filename, encoding='utf8') as file:
        lines = csv.reader(file)
        next(lines) 
        id = 2
        documents = []
        metadatas = []
        ids = []

        for line in lines:
            document = f"Movie Name \"{line[1]}\", subtitle \"{line[2]}\""
            documents.append(document)
            metadatas.append({"Movie Name": line[1]})
            ids.append(str(id))

            if len(ids)>=batch_size:
                queue.put((documents, metadatas, ids))
                documents = []
                metadatas = []
                ids = []

            id+=1

        if(len(ids)>0):
            queue.put((documents, metadatas, ids))


def consumer(use_cuda, queue):
    chroma_client = chromadb.PersistentClient(path="my_vectordb")

    device = 'cuda' if use_cuda else 'cpu'
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="msmarco-bert-base-dot-v5", device= device)
    collection = chroma_client.get_collection(name="sub", embedding_function=sentence_transformer_ef)

    while True:
        batch = queue.get()
        if batch is None:
            break
        
        collection.add(
            documents=batch[0],
            metadatas=batch[1],
            ids=batch[2]
        )

if __name__ == "__main__":

    chroma_client = chromadb.PersistentClient(path="my_vectordb")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="msmarco-bert-base-dot-v5")
 
    try:
        chroma_client.get_collection(name="sub")
        chroma_client.delete_collection(name="sub")
    except Exception as err:
        print(err)

    collection = chroma_client.create_collection(name="sub", embedding_function=sentence_transformer_ef)

    queue = mp.Queue()
    producer_process = mp.Process(target=producer, args=('subtitles.csv', 1000, queue,))
    consumer_process = mp.Process(target=consumer, args=(True, queue,))
    
    start_time = time.time()

    producer_process.start()
    consumer_process.start()
    producer_process.join()
   
    queue.put(None)
    consumer_process.join()

    print(f"Elapsed seconds: {time.time()-start_time:.0f} Record count: {collection.count()}")