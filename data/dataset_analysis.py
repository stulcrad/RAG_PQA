"""Data-quality and corpus-shape report for the PSP speech archive.

Produces the numbers quoted in README §5 (data quality) and §6.4 (why one speech is
one document). Run in the root of the repo, e.g.:

    python data/dataset_analysis.py data/speeches.jsonl

Read-only: it never modifies the corpus.
"""
import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CZ_DOTTED_DATE = re.compile(r"\d{2}\. \d{2}\. \d{4}")

FIELDS = ("id", "speaker", "party", "date", "text")


def load_rows(source: str) -> list[dict[str, Any]]:
    with open(source, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def is_missing(value: Any) -> bool:
    """None, or a string that is empty or whitespace-only."""
    return value is None or (isinstance(value, str) and not value.strip())


def date_format(value: Any) -> str:
    if is_missing(value):
        return "missing"
    if ISO_DATE.fullmatch(value):
        return "ISO (YYYY-MM-DD)"
    if CZ_DOTTED_DATE.fullmatch(value):
        return "Czech dotted (DD. MM. YYYY)"
    return f"unrecognised ({value!r})"


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def report_completeness(rows: list[dict[str, Any]]) -> None:
    section("Field completeness")
    for field in FIELDS:
        missing = sum(1 for row in rows if is_missing(row.get(field)))
        pct = 100 * missing / len(rows)
        flag = "  <- dropped at ingest" if field == "text" and missing else ""
        print(f"  {field:8s} missing/empty: {missing:4d}  ({pct:5.2f}%){flag}")


def report_dates(rows: list[dict[str, Any]]) -> None:
    section("Date formats")
    print("  The brief describes ISO dates; the corpus is mixed.")
    for fmt, count in Counter(date_format(r.get("date")) for r in rows).most_common():
        print(f"  {count:5d}  {fmt}")


def report_duplicates(rows: list[dict[str, Any]]) -> None:
    section("Duplicates")
    print(f"  duplicate ids  : {len(rows) - len({r['id'] for r in rows})}")
    texts = [r["text"] for r in rows if not is_missing(r.get("text"))]
    print(f"  duplicate texts: {len(texts) - len(set(texts))}  ")


def report_lengths(rows: list[dict[str, Any]]) -> None:
    """Justifies the 'one speech = one document, no chunking' decision (README §6.4)."""
    section("Speech length (characters)")
    lengths = sorted(len(r["text"]) for r in rows if not is_missing(r.get("text")))
    def pct(p: float) -> int:
        return lengths[min(int(p * len(lengths)), len(lengths) - 1)]
    print(f"  n={len(lengths)}  min={lengths[0]}  median={int(statistics.median(lengths))}  "
          f"p90={pct(0.90)}  p99={pct(0.99)}  max={lengths[-1]}")
    print(f"  total characters: {sum(lengths):,}")
    print("  bge-m3 accepts 8192 tokens, so even the longest speech fits in one embedding.")


def report_speakers(rows: list[dict[str, Any]]) -> None:
    """Backs the surname-ambiguity limitation (README §5)."""
    section("Speakers")
    speakers = {r["speaker"] for r in rows if not is_missing(r.get("speaker"))}
    print(f"  distinct speakers: {len(speakers)}")

    by_surname: dict[str, set[str]] = defaultdict(set)
    for full_name in speakers:
        by_surname[full_name.split()[-1]].add(full_name)

    shared = {sn: names for sn, names in by_surname.items() if len(names) > 1}
    print(f"  surnames shared by more than one MP: {len(shared)}  "
          f"(a surname alone does not identify a speaker)")
    for surname, names in sorted(shared.items())[:5]:
        print(f"    {surname:12s} -> {', '.join(sorted(names))}")

    prefixes = sorted({(a, b) for a in by_surname for b in by_surname
                       if a != b and b.lower().startswith(a.lower())})
    print(f"  surnames that prefix another surname: {len(prefixes)}  "
          f"(naive substring matching would over-match)")
    for short, long in prefixes[:5]:
        print(f"    {short} / {long}")


def report_retrieval_signals(rows: list[dict[str, Any]]) -> None:
    """Why the speaker is embedded into the document header (README §5)."""
    section("Retrieval signals")
    self_named = sum(1 for r in rows
                     if not is_missing(r.get("speaker"))
                     and r["speaker"].split()[-1] in (r.get("text") or ""))
    print(f"  speeches containing the speaker's own surname: {self_named} / {len(rows)}")
    print("  -> speaker identity is metadata only; it must be embedded into the document")
    print("     header or speaker-scoped questions cannot be answered.")

    for topic in ("digitaliz", "důchod"):
        hits = [r for r in rows if topic in (r.get("text") or "").lower()]
        print(f"  speeches mentioning {topic!r}: {len(hits)}")

    bartos = [r for r in rows if r.get("speaker") == "Ivan Bartoš"]
    on_topic = sum(1 for r in bartos if "digitaliz" in (r.get("text") or "").lower())
    print(f"  Ivan Bartoš: {len(bartos)} speeches, {on_topic} mentioning 'digitaliz'")
    print("  -> a speaker-scoped question about a narrow topic has few relevant speeches;")
    print("     some retrieved context is unavoidably filler.")


def main(source: str) -> None:
    rows = load_rows(source)
    print(f"Corpus report: {source}")
    print(f"Rows: {len(rows)}")
    report_completeness(rows)
    report_dates(rows)
    report_duplicates(rows)
    report_lengths(rows)
    report_speakers(rows)
    report_retrieval_signals(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default="data/speeches.jsonl",
                        help="Path to the speeches JSONL file")
    main(parser.parse_args().source)
