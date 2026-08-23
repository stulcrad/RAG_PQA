"""
PQA interface + RAG implementation: prompt, generation, citation formatting.
"""
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI

from pqa.index import load_index
from pqa.retrieval import build_retriever

PROMPT = """Jsi asistent pro vyhledávání v archivu projevů Poslanecké sněmovny PČR.

# Pravidla pro odpovědi:
- Odpověz na otázku POUZE na základě níže uvedených úryvků z projevů.
- Pokud úryvky neobsahují dostatek informací, výslovně to napiš a nic si nedomýšlej.
- Některé úryvky nemusí s otázkou souviset — ty ignoruj a necituj je.
- Cituj pouze ty úryvky, ze kterých jsi skutečně čerpal, a uveď odkaz přímo za daným tvrzením.
- Každý úryvek začíná jménem řečníka. **Nikdy nepřipisuj výrok jinému řečníkovi, než kdo ho pronesl.**
- Pokud se otázka týká konkrétního řečníka, použij POUZE jeho úryvky a **ostatní ignoruj**.
- U každého tvrzení uveď odkaz na zdroj ve tvaru [1], [2] atd.
- Odpovídej česky, stručně a věcně.

-------

ÚRYVKY:
{context}

OTÁZKA: {question}

ODPOVĚĎ:"""

MAX_DOC_CHARS = 2500 # ~830 tokens; 5 docs ~ 4.2k tokens, increased the cap to 8k tokens

class PQA(ABC):
    """
    An abstract base class for a Political Question Answering (PQA) system.
    """
    @abstractmethod
    def answer(self, query: str) -> str: ...

class RagPQA(PQA):
    """
    A retrieval-augmented generation (RAG) implementation of the PQA system.
    """
    def __init__(self, llm, retriever, k: int = 5):
        self.llm, self.retriever, self.k = llm, retriever, k

    @classmethod
    def from_index(cls, cfg) -> "RagPQA":
        """
        Load persisted index and return an instance of RagPQA.
        Fails loudly if the index hasn't been built yet through the `ingest` command.
        """
        store, docs = load_index(cfg)
        llm = ChatOpenAI(model=cfg.llm_model, base_url=cfg.base_url,
                         api_key=cfg.api_key, temperature=0.0) # deterministic answers
        retriever = build_retriever(store, docs, cfg.top_k, hybrid=cfg.hybrid)
        return cls(llm, retriever, cfg.top_k)

    def answer(self, query: str) -> str:
        """
        Answer a natural language query using the RAG approach.

        Args:
            - query: The user's natural-language question.

        Returns:
            - A natural-language answer including references to the source speeches where possible.
        """
        docs = self.retriever.invoke(query)[:self.k]
        context = self._format_context(docs)
        text = self.llm.invoke(PROMPT.format(context=context, question=query)).content
        return f"{text}\n\n{self._format_sources(docs)}"

    @staticmethod
    def _format_context(docs) -> str:
        """
        Format the retrieved documents into a context string for the prompt.

        Args:
            - docs: A list of Document objects retrieved by the retriever.
        Returns:
            - A string containing the formatted context for the prompt.
        """
        return "\n\n".join(
            [f"[{i+1}] {d.page_content[:MAX_DOC_CHARS]}" for i, d in enumerate(docs)]
            )

    @staticmethod
    def _format_sources(docs) -> str:
        """
        Format the sources of the retrieved documents into a string for citation.

        Args:
            - docs: A list of Document objects retrieved by the retriever.
        Returns:
            - A string containing the formatted sources for citation.
        """
        lines = ["ZDROJE:"]
        for i, d in enumerate(docs):
            m = d.metadata
            lines.append(f"[{i+1}] {m.get('speaker') or 'neznámý řečník'} ({m.get('party') or 'neznámá strana'}, "
                         f"{m.get('date') or 'neznámé datum'}) - id={m['id']}")
        return "\n".join(lines)
                         