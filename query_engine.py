class QueryEngine:
    """Handles querying the retriever."""

    def __init__(self, retriever):
        self.retriever = retriever

    def query(self, query_text):
        results = self.retriever.invoke(query_text)

        print("\nTop retrieved documents:\n")
        for i, doc in enumerate(results[:3]):
            print(f"--- Document {i+1} ---")
            print(doc.page_content[:300], "\n")

        return results
