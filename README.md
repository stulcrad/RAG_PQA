# Parliamentary Q&A (PSP) — RAG Proof of Concept

Natural-language Q&A over ~2,500 speeches from the Poslanecká sněmovna archive. Ask a
question in Czech, get an answer grounded in retrieved speeches with references back to
the source records.

Everything runs **on-prem**. The LLM and the embedding model are served behind an
**OpenAI-compatible endpoint** (Ollama here) and reached through the standard OpenAI
client; **no API key and no external network calls**.

---

## Project layout

```
cli.py                  argparse only; delegates to pqa/
evaluation.py           retrieval evaluation + dense/hybrid ablation
metrics.py              recall@k, MRR, latency percentiles (pure functions)
Modelfile               gemma3-pqa: gemma3:4b-it-qat with num_ctx 8192
pqa/
  config.py             .env -> frozen Config dataclass, with working defaults
  corpus.py             JSONL -> Documents; date normalisation, empty-text dropping, headers
  index.py              embeddings + FAISS build/load; writes docs.jsonl and meta.json
  retrieval.py          build_retriever() — the dense/hybrid seam
  qa.py                 PQA ABC + RagPQA: prompt, generation, citation formatting
data/
  speeches.jsonl        input corpus
  eval.jsonl            17 hand-authored eval cases with verified gold IDs
  dataset_analysis.py   corpus statistics / data-quality checks
```

---

## 1. Quickstart

### Prerequisites

```bash
# Linux / WSL2 (tested on Ubuntu 22.04)
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# 2. Models
ollama pull bge-m3            # embeddings, multilingual  (~1.2 GB)
ollama pull gemma3:4b-it-qat  # generation               (~2.6 GB)

# 3. Generation model with an enlarged context window (see §6.5)
ollama create gemma3-pqa -f Modelfile

# 4. Isolated Python environment (Python 3.11+; developed on 3.13)
python -m venv .venv
source .venv/bin/activate
# or with conda:  conda create -n pqa python=3.13 pip && conda activate pqa

# 5. Dependencies
pip install -r requirements.txt     # exact pinned versions (reproduces the §2 measurements)

# Optional: editable install for development, no deps (already installed above)
pip install -e . --no-deps
```

### Run

```bash
# Index the corpus (~26 min on the reference hardware, see §3)
python cli.py ingest data/speeches.jsonl

# Ask questions
python cli.py answer "Co si myslí Bartoš o digitalizaci státní správy?"
python cli.py answer "Co poslanci diskutovali k důchodové reformě?"
```

Useful flags:

```bash
python cli.py ingest data/speeches.jsonl --limit 50   # fast smoke test of the whole path
python cli.py answer "..." --hybrid                   # hybrid BM25+dense (off by default, see §4)
```

Progress and diagnostics go to **stderr**; the answer goes to **stdout**, so
`python cli.py answer "..." > out.txt` yields a clean file.

### Configuration

All settings come from `.env`, with working defaults so the tool runs with no `.env` at all.

| Variable            | Default                       | Purpose                                       |
| ------------------- | ----------------------------- | --------------------------------------------- |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | on-prem endpoint; point at vLLM in production |
| `OPENAI_API_KEY`  | `ollama`                    | any non-empty string; no real key is used     |
| `LLM_MODEL`       | `gemma3:4b-it-qat`          | set to`gemma3-pqa` after step 3 above       |
| `EMBEDDING_MODEL` | `bge-m3`                    | embedding model for dense retrieval           |
| `INDEX_DIR`       | `.index`                    | where FAISS artifacts land                    |
| `TOP_K`           | `5`                         | speeches passed to the LLM                    |
| `HYBRID`          | `false`                     | enable BM25+dense fusion                      |

---

## 2. Evaluation

```bash
python evaluation.py                # dense (configured default)
python evaluation.py --hybrid       # hybrid BM25+dense
python evaluation.py --ablation     # both, side by side
```

Evaluation is **retrieval-only** and deliberately so: it needs no LLM, runs in seconds,
and does not compete with the generation model for VRAM. Retrieval is also the ceiling on
answer quality — no prompt can recover a speech that was never retrieved.

The eval set is `data/eval.jsonl`, **17 LLM-generated questions, checked by hand against the corpus** whose gold speech IDs
were located by searching the corpus for distinctive terms (`kurzarbeit`, `Turów`,
`Čapí hnízdo`, `chytrá karanténa`, …) that appear in only 1–3 speeches, so the gold label
is unambiguous.

### Results

Reference hardware: **NVIDIA GTX 1650 (4 GB VRAM), 8 CPU cores, 7 GB RAM, WSL2**.

| Stage                       | Measurement                                                       |
| --------------------------- | ----------------------------------------------------------------- |
| Ingest                      | 2,475 speeches in**1,589 s (26.5 min)**, ~0.6–0.8 s/speech |
| Embedding dimension         | 1,024 (`bge-m3`)                                                |
| Rows dropped at ingest      | 25 (empty`text`) out of 2,500                                   |
| **recall@5**          | **0.784**                                                   |
| **MRR@5**             | **0.716**                                                   |
| Retrieval latency (warm)    | p50**~118 ms**, p95 ~245 ms                                 |
| End-to-end`answer` (warm) | **~79 s**                                                   |

### What the numbers mean

**recall@5 = 0.784** — averaged over the 17 cases, 78 % of the gold speeches appear in the
top 5 retrieved. Since most cases have exactly one gold speech, this is close to "the right
speech is in the context roughly 4 times out of 5."

**MRR@5 = 0.716** — the first correct speech sits at rank ~1.4 on average. Retrieval isn't
merely finding the right speech, it's ranking it at or near the top, which matters because
the LLM attends unevenly across a long context.

**~118 ms retrieval vs ~79 s end-to-end** — **essentially all latency is generation**, on a
4 GB card running a 4B model with an 8k context. On PSP's own hardware this collapses.
Optimise the serving stack, not the retriever.

**26.5 min ingest** is a one-off batch cost, single-threaded against one local endpoint. A
production run would batch across replicas.

⚠️ **Read these as indicative, not authoritative.** n=17 is small, and gold labels come from
term matching rather than human relevance judgement. The *relative* comparison between
retrieval configurations (§4) is the trustworthy signal; the absolute values are not.

---

## 3. Worked example — verifying the citations

The point of this tool is sourcing, so here is one full answer traced back to the corpus.

```
$ python cli.py answer "Co si myslí Bartoš o digitalizaci státní správy?"
```

> Bartoš si myslí, že důkladnou analýzou již teď se na celkových nákladech dá výrazně
> ušetřit **[2]**. Důležité je, aby systémy, ke kterým nemá nikdo klíče, nebyly budovány za
> vysoké ceny **[2]**. Navíc, regionální rozvoj je podporován prostřednictvím programů jako
> IROP nebo kohezní fondy **[2]**. V minulosti vláda neprosadila žádný viditelný výsledek
> v digitalizaci státní správy **[3]**. Ačkoli se to zdá být vnímáno jako útok, je to často
> věcná kritika, kterou Bartoš odmítá **[3]**. **[4]**
>
> ```
> ZDROJE:
> [1] Ivan Bartoš (Piráti, 2018-07-11) - id=ParlaMint-CZ_2018-07-11-ps2017-017-01-001-001.u27
> [2] Ivan Bartoš (Piráti, 2022-02-18) - id=ParlaMint-CZ_2022-02-18-ps2021-011-01-002-002.u153
> [3] Ivan Bartoš (Piráti, 2018-01-10) - id=ParlaMint-CZ_2018-01-10-ps2017-005-01-001-001.u20
> [4] Ivan Bartoš (Piráti, 2022-02-18) - id=ParlaMint-CZ_2022-02-18-ps2021-011-01-002-002.u147
> [5] Ivan Bartoš (Piráti, 2021-05-12) - id=ParlaMint-CZ_2021-05-12-ps2017-102-01-001-999.u20
> ```

All five retrieved speeches are genuinely Ivan Bartoš — dense retrieval keeps the answer
inside the right speaker. Checking each claim against the raw corpus:

| Claim                                                          | Cited | Source text                                                                                                                                                            | Verdict                      |
| -------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| analýzou lze ušetřit na nákladech                          | [2]   | *"…důkladnou analýzou již teď se na celkových nákladech dá výrazně ušetřit."*                                                                            | ✅ faithful                  |
| systémy, ke kterým nemá nikdo klíče                       | [2]   | *"…historicky se budovaly systémy, ke kterým nemá nikdo klíče, pouze dodavatel, kde se platí obrovské sumy za licence…"*                                    | ✅ faithful                  |
| regionální rozvoj přes IROP / kohezní fondy                | [2]   | *"…regionální rozvoj, investice do regionů je pokrývána právě skrze často třeba kohezní fondy nebo přes IROP…"*                                         | ✅ faithful                  |
| za minulé vlády žádný viditelný výsledek v digitalizaci | [3]   | *"Hnutí ANO bylo ve vládě čtyři roky a v resortech, které mělo ve své gesci, nebyl prosazen žádný viditelný výsledek v digitalizaci státní správy."* | ✅ faithful                  |
| *"věcná kritika, kterou Bartoš odmítá"*                 | [3]   | *"Tam, kde tedy přicházela věcná kritika, tam byl pan Babiš v pozici ministra hluchý. Věcná kritika je často chápána … jako útok."*                     | ❌**meaning inverted** |
| trailing bare**[4]**                                           | [4]   | — no claim attached to it                                                                                                                                             | ⚠️ dangling citation       |

**Four of five substantive claims are faithful. The fifth inverts its source.** The speech
says that *substantive criticism was ignored by Babiš and is often perceived as an attack*;
Bartoš is the one **making** the criticism. The generated sentence turns him into the person
**rejecting** it. Every individual word can be traced to the cited speech; the relationship
between them is wrong.

This is a more dangerous failure than a wrong citation number. A reviewer spot-checking `[3]`
finds matching vocabulary and may accept the sentence. Only reading the passage in full
reveals the stance has been reversed — and reversing an MP's position is precisely the error
a parliamentary archive tool cannot afford. The trailing bare `[4]`, a citation with no
claim attached, is the same weakness in milder form.

This example is why §7 lists automated answer-level evaluation as the first priority: no
retrieval metric in §2 can detect either defect.

---

## 4. Retrieval ablation — why dense-only

Hybrid BM25+dense retrieval was implemented, measured, and then **disabled by default**.
Reproduce with:

```bash
python evaluation.py --ablation
```

```
retrieval                       recall@5    mrr@5    p50 s
dense                              0.784    0.716   0.1067
hybrid (bm25+dense) [0.7, 0.3]     0.784    0.716   0.1624
hybrid (bm25+dense) [0.5, 0.5]     0.745    0.683   0.1352
hybrid (bm25+dense) [0.3, 0.7]     0.078    0.118   0.1446
hybrid (bm25+dense) [0.0, 1.0]     0.078    0.044   0.1569
```

(Weights are `[dense, bm25]`; the last row is BM25 alone.)

**BM25 is near-useless on this corpus, and the reason is Czech morphology.**
`BM25Retriever`'s default tokenizer is `str.split()` — no lowercasing, no stemming,
punctuation left attached.

```python
>>> retriever.preprocess_func("Co zaznělo o kotlíkových dotacích?")
['Co', 'zaznělo', 'o', 'kotlíkových', 'dotacích?']
```

Czech is heavily inflected, so the query token `kotlíkových` never matches the corpus form
`kotlíkové`; `kurzarbeitu` never matches `kurzarbeit`.
Lexical retrieval degenerates to noise.

The weight sweep shows there is no setting where BM25 helps. At `[0.5, 0.5]` its noise
**costs 4 points of recall**. At `[0.7, 0.3]` its contribution vanishes into the dense
ranking entirely — identical scores to dense, so it buys nothing while adding a dependency
and ~50 ms of latency. Below that, it actively dominates and recall collapses to 0.078.

Hybrid also caused a **cross-speaker misattribution**: for the Bartoš query it pulled in
speeches by Ondřej Profant and Václav Klaus (BM25 matched the topic term across all
speakers), and the LLM then sourced its entire answer from a Profant speech while
attributing it to Bartoš. Dense-only does not exhibit this on the same query (§3).

The code is kept behind `--hybrid` / `HYBRID=true`, with the weights parameterised, so the
comparison stays reproducible rather than being a claim in a document. **This is not an
argument against hybrid retrieval in general**; it is an argument that BM25 needs
Czech-aware tokenisation before it can contribute (§7).

---

## 5. Known limitations

**Answer quality is not evaluated.** This is the biggest gap. Metrics cover retrieval only;
nothing automatically checks whether the generated answer is faithful to its sources. The
numbers in §2 are retrieval ceilings, not answer quality.

**Answers can distort their sources.** Vocabulary is drawn faithfully from the cited speech,
but the relationship between terms is not always preserved. §3 contains a sentence that
reverses an MP's stance while citing the correct speech, plus a dangling citation with no
claim attached. If the context also contained a speech by a different MP, the LLM have
misattributed the claim to the wrong speaker entirely.
Prompt-level instructions ("cite only what you used", "never attribute across speakers")
were tried and did **not** fix any of this; a 4B model treats such rules as suggestions.
Constraints that must hold need to be enforced in code, or detected by evaluation.

**Speaker identity competes with topic in the embedding.** Each document is embedded with a
header (`Řečník: Ivan Bartoš (Piráti), 2022-02-18`), which is what makes speaker-scoped
questions work at all -> the speaker's name appears in their own speech text in only 9 of
2,500 records (proof below). But it also means identity and topic share one vector and compete: a query
naming an MP retrieves that MP's speeches about *anything*. Bartoš has 12 speeches, only 3
mentioning digitalisation, so some context slots are always filler. The right fix is
structural filtering (§7), not a better embedding.

**Surnames are ambiguous.** 15 surnames in the corpus belong to more than one MP —
`Fiala` → Petr Fiala *and* Radim Fiala; three different `Holeček`s; three `Černý`s. "Co si
myslí Fiala o X" is genuinely under-specified and the system has no disambiguation step.

**Long speeches are truncated in the prompt.** Documents are capped at
`MAX_DOC_CHARS = 2500` when building the LLM context (p99 speech length is 4,869 chars,
max 6,440). The relevant passage can fall outside the cap. This is the cost of not chunking
(§6.4).

**The input data does not match its stated contract.** The brief describes ISO dates and a
cleaned corpus. Actual findings across the 2,500 rows:

| Issue                                    | Count |
| ---------------------------------------- | ----- |
| Empty`text` (dropped at ingest)        | 25    |
| Dates as`01. 03. 2017` rather than ISO | 25    |
| Missing`date`                          | 13    |
| Missing`speaker`                       | 18    |
| Missing/empty`party`                   | 277   |

Dates are normalised at ingest and empty texts dropped. Among the 2,475 usable speeches there
are **no duplicate texts and no duplicate IDs**

A deeper analysis of the dataset can be run with `python data/dataset_analysis.py`, from
which many limitations above were discovered.

---

## 6. Design decisions

### 6.1 RAG

The brief requires answers *"including references to the source speeches"*. Citations
require retrieval by construction, so **retrieval-augmented generation (RAG)** is an
obvious choice.

### 6.2 Ollama, reached through the OpenAI client

The brief allows any OpenAI-compatible on-prem endpoint. vLLM is the better choice for
production throughput on datacentre GPUs, but it pre-allocates a large KV cache and will not
fit the 4 GB reference card; Ollama quantises to GGUF and offloads to CPU when VRAM runs out.

You can change the endpoint by setting `OPENAI_BASE_URL` in `.env`, and the client will continue to
work. The code (`ChatOpenAI`) is written against the OpenAI client, so a future swap to vLLM or any other OpenAI-compatible endpoint is a one-line change in `pqa/config.py`.

### 6.3 Models: `bge-m3` and `gemma3:4b-it-qat`

**`bge-m3`** — the corpus is Czech. `nomic-embed-text`, named in the brief, is
English-centric. `bge-m3` is genuinely multilingual, strong on Czech, 1024-dim, 8k context,
and **symmetric** — no `query:` / `passage:` prefixes to mismatch (unlike `multilingual-e5`).

**`gemma3:4b-it-qat`** — good Czech generation at a size that fits 4 GB. The QAT build is
~2.6 GB versus ~3.3 GB for the standard 4-bit quantisation. Smaller models degrade sharply
on Czech grammar. This model is the main quality bottleneck (§5, §7).

### 6.4 One speech = one document (no chunking)

Speech lengths (2,475 usable): min 200, median 628, p90 2,022, p99 4,869, max 6,440. `bge-m3` handles 8,192
tokens, so even the longest speech fits in a single embedding. Chunking would have added
code and forced a decision about citation granularity. Worth doing only if the prompt truncation proves to
be a real problem, which we cannot know without answer-level evaluation (§7).

The trade-off is the 2,500-char prompt truncation in §5 — a long speech's relevant passage
can fall outside the cap.

**Two different fixes.** Raising `num_ctx` (e.g. to 10k) and the truncation cap (to ~3,500 chars) 
covers the p99 speech (4,869 chars) almost entirely, for zero code complexity — the cheap fix, 
worth doing first. However, it does not fully solve it: the max observed speech is 6,440 chars, 
so the cap still has to sit somewhere below that, and every extra token of window cost is paid on 
**every** query regardless of whether that query needed it. Parent-document retrieval 
(embed passages, cite the parent speech) is the fix that scales with corpus size instead of 
with the longest outlier, at the cost of actual implementation (§7). I would 
Start with the cheap fix; reach for the structural one only if it stops being enough.

### 6.5 Context window: why `gemma3-pqa` exists

Ollama defaults `num_ctx` to **4,096 tokens**. Five speeches plus instructions measured at
**~6,100 tokens** for the Bartoš query. Ollama does not error on overflow — it **truncates
from the front**, which is exactly where the instructions live. The model would silently
lose "answer only from the excerpts", "cite your sources", "answer in Czech" and keep the
raw speeches, producing plausible but ungoverned output.

`Modelfile` raises this to 8,192:

```
FROM gemma3:4b-it-qat
PARAMETER num_ctx 8192
```

Combined with `MAX_DOC_CHARS = 2500`, the prompt should now fit together with sources intact. 

### 6.6 FAISS, not a vector database

2,475 vectors × 1,024 dims is a **10 MB matrix**. FAISS `IndexFlatL2` brute-forces it
exactly in microseconds — no approximation error, and no server for PSP to deploy before they can
evaluate the PoC. Milvus or Qdrant become the right answer at the scale of the full archive
(millions of speeches); `build_retriever` is the seam where that swap happens.


### 6.7 LangChain

Used for `FAISS`, `BM25Retriever`, `EnsembleRetriever` and the OpenAI-compatible clients;
roughly the retrieval plumbing and nothing else. The ablation in §4 is a few lines precisely
because swapping retrievers is a one-line change behind a shared `BaseRetriever` interface.

### 6.8 `temperature=0.0`

Greedy decoding removes sampling noise, so metric changes can be attributed to code changes
rather than to chance.

---

## 7. Next steps

Roughly in order my own priorities if I had more time would be:

**1. Evaluate the generated answers.** The largest gap. Three tractable metrics:

- *Groundedness / faithfulness* — LLM-as-judge scoring whether each cited claim is supported
  by the speech it cites. Would have caught the §3 miscitation automatically.
- *Attribution precision* — for speaker-scoped questions, the fraction of cited sources
  actually spoken by the named MP. Needs no LLM and directly targets the §4 misattribution.
- *Abstention rate* — how often the system correctly declines when context is insufficient.

**2. Structural speaker filtering.** When a question names an MP, filter retrieved documents
by the `speaker` metadata instead of hoping the LLM respects the instruction.
Match known surnames in the query (prefix matching to tolerate Czech
declension — `Bartoše`, `Bartošovi`), resolving ambiguity **per match position** so
`Bartoš a Babiš` keeps both while `Bartošek` does not also match `Bartoš`, and retaining all
MPs who share a surname so the LLM can disambiguate from the headers.

**3. Query understanding.** Generalises step 2: have the LLM extract a structured filter
(`speaker`, `party`, `date_range`) from the question and apply it as metadata pre-filtering
before vector search, rather than blending identity and topic into one embedding. Also the
place to ask the user which `Fiala` they mean.

**4. Czech-aware BM25, then revisit hybrid.** Supply a custom `preprocess_func` with lowercasing,
diacritics handling and Czech stemming/lemmatisation (e.g. `simplemma`). Lexical retrieval
should genuinely help for surnames, party names and bill numbers once morphology is handled.

**5. A larger generation model.** The 4B model is the binding constraint on citation
discipline. On better hardware a 27B-class model (or a hosted on-prem 70B) should sharply
reduce both miscitation and the 79 s latency.

**6. Chunking with parent-document retrieval.** Embed passages for precise matching, return
the parent speech for citation. Removes the `MAX_DOC_CHARS` truncation and stops long
speeches from being retrieved on the strength of one paragraph, is in contradiction with §6.4, 
and is the right fix if the cheap fix stops being enough.

**7. A larger, auto-generated eval set.** Prompt the LLM to write questions answerable only
from a held-out speech; gold = that speech ID.

**8. A test suite.** There is currently none. The pure-logic pieces are the cheap, high-value targets, all testable without an LLM or an index, e.g.:

- `normalize_date` — ISO input, Czech dotted input, missing, and unparseable-passthrough.
- `metrics.recall_at_k` / `reciprocal_rank` — gold at rank 1, at rank k, absent, multiple golds.
- `corpus.load_speeches_from_jsonl` — the drop count, and that `None` speaker/party never
  reaches the header as the literal string `"None"` (a bug that did ship once).
- `speakers_in_query` (once §2 lands) — `Bartoš a Babiš` keeps both, `Bartošek` does not match `Bartoš`.

Retrieval and generation need fixtures and are better covered by the evaluation harness than
by unit tests.

---

## 8. Note on time

The brief suggested approximately two hours. My actual effort was longer: 
roughly two hours went to environment setup — installing Ollama, downloading ~4.5 GB of models,
building the repo structure, studying documentations, and writing this README. Analysis of the dataset, 
writing the code, and running the evaluation harness took another 2 hours.

I understand that the brief was a time-boxed exercise, and I have spent more time than expected.
In the future I would have cut the time by having a prepared project template, having the models 
pre-downloaded, knowing already how to use libraries like LangChain, and spending less time on the README,
which I think is too long. For a PoC, I would have cut §4, §5, §6, and §7, and just reported the retrieval metrics
and offered next steps to the client. 
