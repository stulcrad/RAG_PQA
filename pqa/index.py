"""Embedding + FAISS index build/load. Artifacts live in config.index_dir."""

import json
import sys
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from pqa.corpus import load_speeches_from_jsonl


def make_embeddings(cfg) -> OpenAIEmbeddings:
    """
    Embedding client for the on-prem OpenAI-compatible endpoint.
    
    Args:
        - cfg: Configuration object containing parsed env variables.
    Returns:
        - An instance of OpenAIEmbeddings configured with the specified model and parameters.
    """
    return OpenAIEmbeddings(
        model=cfg.embedding_model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        check_embedding_ctx_length=False,
        chunk_size=32,
    )


def build_index(cfg, source: str, limit: int | None = None) -> None:
    """
    Embed the corpus and persist FAISS index + docs.jsonl to cfg.index_dir.

    Args:
        - cfg: Configuration object containing parsed env variables.
        - source: Path to the JSONL file containing speeches.
        - limit: Optional integer to limit the number of speeches to ingest.
    """
    docs, drop_count = load_speeches_from_jsonl(source)
    # Log the number of dropped documents to stderr
    print(f"Loaded {len(docs)} documents from {source}, dropped {drop_count} empty text entries.", file=sys.stderr)

    if limit:
        docs = docs[:limit]

    # Make embeddings for all docs, in batches of 32, and store them in a FAISS index
    emb = make_embeddings(cfg)
    texts = [d.page_content for d in docs]
    vectors, t0 = [], time.perf_counter()

    for i in range(0, len(texts), 32):
        vectors.extend(emb.embed_documents(texts[i:i + 32]))
        # progress -> stderr, so stdout stays clean for the `answer` contract
        print(f"  embedded {min(i+32, len(texts))}/{len(texts)}", file=sys.stderr)
    elapsed = time.perf_counter() - t0

    store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=emb,
        metadatas=[d.metadata for d in docs],
        normalize_L2=True, # FAISS recommends this for cosine similarity search
    )

    # Save the FAISS index and the documents to the specified index directory
    d = cfg.index_dir; d.mkdir(parents=True, exist_ok=True)
    store.save_local(str(d))
    # BM25Retriever needs raw docs later; don't reach into store.docstore._dict
    with open(d / "docs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"page_content": doc.page_content,
                                "metadata": doc.metadata}, ensure_ascii=False) + "\n" for doc in docs)
    
    # ingest stats for the README / PSP's "performance results"
    (d / "meta.json").write_text(json.dumps({
        "n_docs": len(docs), "n_dropped": drop_count,
        "embedding_model": cfg.embedding_model,
        "dim": len(vectors[0]), "ingest_seconds": round(elapsed, 1),
    }, indent=2))


def load_index(cfg) -> tuple[FAISS, list[Document]]:
    """
    Load persisted index + docs. Raises if ingest hasn't run.

    Args:
        - cfg: Configuration object containing parsed env variables.
    Returns:
        - A tuple containing:
            - An instance of FAISS loaded from the specified index directory.
            - A list of Document objects loaded from docs.jsonl.
    
    """
    # Load the documents from docs.jsonl
    docs_path = Path(cfg.index_dir) / "docs.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"docs.jsonl not found in {cfg.index_dir}. Please run ingest first.")
    
    store = FAISS.load_local(str(cfg.index_dir), make_embeddings(cfg),
                             allow_dangerous_deserialization=True)
    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            doc_data = json.loads(line)
            docs.append(Document(page_content=doc_data["page_content"], metadata=doc_data["metadata"]))

    return store, docs

