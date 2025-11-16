from .answer_relevance import score_answer_relevance
from .context_relevance import score_context_relevance
from .groundedness import score_groundedness

def evaluate_answer(question, answer, reference_answers, contexts, sources):
    """
    Evaluates the answer using three metrics.
    
    Args:
        question: The user's question
        answer: The generated answer
        reference_answers: List of expected/correct answers
        contexts: Retrieved context chunks
        sources: Source documents
    
    Returns:
        Dictionary with three evaluation scores
    """
    return {
        "answer_relevance": score_answer_relevance(question, answer, reference_answers),
        "context_relevance": score_context_relevance(answer, contexts),
        "groundedness": score_groundedness(answer, sources)
    }