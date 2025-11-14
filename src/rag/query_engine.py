class QueryEngine:
    """Handles similarity search and retrieval."""

    def __init__(self, retriever):
        self.retriever = retriever

    def query(self, query_text):
        results = self.retriever.get_relevant_documents(query_text)
        print(f"\nQuery: {query_text}\n")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:\n{doc.page_content[:400]} ...")
        return results