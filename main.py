from data_loader import DataLoader
from chunker import TextChunker
from embedder import Embedder
from vectorstore_manager import VectorStoreManager
from query_engine import QueryEngine

if __name__ == "__main__":
    text_files = [
        "cleaned_texts/automate-the-boring-stuff-with-python-3rd-edition-early-access-3nbsped-9781718503403-9781718503410_com (1).txt",
        "cleaned_texts/fluent-python-2nbsped-9781492056348-9781492056287.txt",
        "cleaned_texts/learning-python-powerful-object-oriented-programming-6nbsped-1098171306-9781098171308.txt",
        "cleaned_texts/leetcode-python.txt",
        "cleaned_texts/python-crash-course-3nbsped-1718502702-9781718502703_compress.txt"
    ]

    loader = DataLoader(text_files)
    documents = loader.load()

    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(documents)

    embedder = Embedder()
    embedding_model = embedder.get_model()

    vector_manager = VectorStoreManager(embedding_model)
    qdrant_store = vector_manager.create_store(chunks)

    retriever = vector_manager.get_retriever(top_k=5)
    query_engine = QueryEngine(retriever)
    query_engine.query("What is recursion?")
