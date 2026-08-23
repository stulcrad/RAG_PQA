# Assignment: Parliamentary Q&A — Proof of Concept

## Background

Your team has been approached by the **Poslanecká sněmovna Parlamentu ČR (PSP)** to build a
proof-of-concept Q&A tool over their parliamentary speech archive. The parliament wants to allow
researchers, journalists, and parliamentary staff to ask questions in natural language and
receive relevant, sourced answers.

The PoC will be demonstrated to **PSP's IT department**, who will assess whether it is worth
investing in a full solution. PSP employees will test the PoC on their own. They will also ask
for performance results. Code quality and architecture matter — this is not a throwaway script.

## Instructions

You have approximately **2 hours**. Make sure you deliver working code - but metrics don't have to
be optimal. You are encouraged to note next steps/TODOs for later discussion. Feel free to use AI,
but make sure you understand the code, functionality and core decisions.

---

## What You Are Given

**`speeches.jsonl`** — A pre-processed, cleaned subset of ~2,500 speeches from PSP open data from
  the client (speaker, party, text, ISO date).

---

## What We Expect

Deliver a Python **working CLI tool** that allows a user to first ingest the corpus and then query it.

The following commands **must work exactly as written**:

```bash
# Index the corpus from a JSONL file
python cli.py ingest data/speeches.jsonl

# Query the system
python cli.py answer "Co si myslí Bartoš o digitalizaci státní správy?"
python cli.py answer "Co poslanci diskutovali k důchodové reformě?"
```

The `answer` command should print a natural language response to stdout, including references to
source speeches where possible.

---

## Notes on Scope

**Language and embedding models:** Both the LLM and the embedding model are served on-prem
behind an **OpenAI API-compatible endpoint** (e.g. vLLM, Ollama, ...). Your code should talk to
them via the standard OpenAI client. The endpoint should run on-prem, so **no API key shall**
**be required** - feel free to use local small language models (e.g. Gemma, Nomic Embed, ...).

**Perfection is not the goal.** A tractable, reasoned solution with clear trade-off awareness
is far more valuable than a polished but opaque one. Trade-offs, decisions etc. will be
discussed in person, but it is recommended to include them in your `README.md`.

---

## Example Interface

```python
from abc import ABC, abstractmethod


class PQA(ABC):

    @abstractmethod
    def answer(self, query: str) -> str:
        """
        Answer a natural language query.

        Args:
            query: The user's natural-language question.

        Returns:
            A natural-language answer including references to the source
            speeches where possible.
        """
        ...
```
