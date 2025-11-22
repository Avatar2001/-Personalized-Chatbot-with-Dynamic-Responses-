from sentence_transformers import CrossEncoder
from typing import List

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cpu", max_length=512):

        self.model = CrossEncoder(
            model_name_or_path=model_name,
            device=device,
            max_length=max_length
        )

    def rerank(self, query: str, docs:List[str],top_k: int=5):
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs, batch_size=16)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return top