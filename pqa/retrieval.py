"""
A file containing the logic to build a retrieval-based QA system using LangChain.
"""
from collections.abc import Sequence

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever


def build_retriever(vectorstore, docs, k: int, hybrid: bool = False, 
                    weights: Sequence[float] = [0.7, 0.3]) -> BaseRetriever:
    """
    Build a retriever that can be used to retrieve relevant documents for a given query.
    Dense by default. BM25's whitespace tokenizer does not handle Czech inflection 
    (recall@5 0.078 vs 0.784 dense) — see README ablation.
    
    Args:
        - vectorstore: The vectorstore to use for dense retrieval.
        - docs: The documents to use for BM25 retrieval.
        - k: The number of documents to retrieve.
        - hybrid: Whether to use hybrid retrieval (BM25 + dense) or dense-only.
        - weights: The weights to use for the ensemble retriever (only used if hybrid is True).
    Returns:
        - A retriever that can be used to retrieve relevant documents for a given query.
    """
    dense = vectorstore.as_retriever(search_kwargs={"k": k})
    if not hybrid:
        return dense
    bm25 = BM25Retriever.from_documents(docs, k=k)
    return EnsembleRetriever(retrievers=[dense, bm25], weights=weights)
