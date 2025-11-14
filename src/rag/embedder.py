from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    """Creates embeddings for documents or queries."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        print(f"Embedding model '{model_name}' loaded.")

    def embed_query(self, text):
        return self.model.embed_query(text)

    def get_model(self):
        return self.model