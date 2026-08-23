"""
A main file that will be used to run the CLI for the QA system.

This is argparse only, the actual logic is in the pqa module. The CLI will work as follows:

# Index the corpus from a JSONL file
python cli.py ingest data/speeches.jsonl
# Query the system
python cli.py answer "Co si myslí Bartoš o digitalizaci státní správy?"

Answer to the stdout will be a natural-language answer including references to the source speeches where possible.

The CLI will use the default model specified in the .env file or default to the default configuration
in pqa/config.py if no .env file is present.
"""
import argparse
from dataclasses import replace

from pqa.config import load_config
from pqa.index import build_index
from pqa.qa import RagPQA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Political Question Answering CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a source of speeches")
    ingest_parser.add_argument("source", type=str, help="Path to the source of speeches")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Limit the number of speeches to ingest")

    answer_parser = subparsers.add_parser("answer", help="Answer a natural language query")
    answer_parser.add_argument("query", type=str, help="The user's natural-language question")
    answer_parser.add_argument("--hybrid", action="store_true",
                               help="Use hybrid BM25+dense retrieval instead of dense-only "
                                    "(scores worse on Czech - see README ablation)")

    args = parser.parse_args()

    cfg = load_config()  # Load configuration from .env file

    if args.command == "ingest":
        build_index(cfg, args.source, args.limit)
    elif args.command == "answer":
        if args.hybrid:
            cfg = replace(cfg, hybrid=True)   # CLI flag overrides .env
        ragpa = RagPQA.from_index(cfg)
        response = ragpa.answer(args.query)
        print(response)
