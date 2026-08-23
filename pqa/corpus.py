"""
A file to load the jsonl speeches, normalize dates, drop empty text, and build Documents
"""
import json
from datetime import datetime
from typing import Any

from langchain_core.documents import Document

# Formats observed in the corpus: mostly ISO, 25 rows in Czech dotted notation.
DATE_FORMATS = ("%Y-%m-%d", "%d. %m. %Y")


def load_speeches_from_jsonl(file_path: str) -> tuple[list[Document], int]:
    """
    Load speeches from a JSONL file and return a list of Document objects.

    Args:
        - file_path: The path to the JSONL file containing speeches.

    Returns:
        - A tuple containing:
            - A list of Document objects created from the speeches.
            - An integer representing the number of speeches dropped due to empty text.
    """
    documents = []
    drop_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            speech = json.loads(line)
            if speech.get("text"): # Skip entries without text
                document = create_document_from_speech(speech, file_path)
                documents.append(document)
            else:
                drop_count += 1
    return documents, drop_count


def normalize_date(date_str: str) -> str:
    """
    Normalize a date string to the ISO format YYYY-MM-DD if possible, otherwise return the original string.

    Args:
        - date_str: The date string to normalize.

    Returns:
        - A normalized date string in ISO format or the original string if it cannot be parsed.
    """
    if not date_str:
        return ""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).date().isoformat()
        except ValueError:
            continue
    return date_str  # unrecognised format: keep the original rather than losing information

def format_header(speaker, party, date) -> str:
    """
    Format the header for a speech document.

    Args:
        - speaker: The name of the speaker.
        - party: The political party of the speaker.
        - date: The date of the speech.

    Returns:
        - A formatted string representing the header of the speech document.
    """
    who = speaker or "neznámý řečník"
    if party:
        who += f" ({party})"
    if date:
        who += f", {date}"
    return f"Řečník: {who}\n\n"

def create_document_from_speech(speech: dict[str, Any], file_path) -> Document:
    """
    Create a Document object from a speech dictionary.
    The Document's content will be a header of the form: 
    "Řečník: {speaker} ({party}), {date}\n\n{speech_text}"

    Metadata will include the source file name speech id, speaker, party, and date.

    Args:
        - speech: A dictionary containing speech data.
        - file_path: The path to the source JSONL file.
    Returns:
        - A Document object representing the speech.
    """
    id, speaker, party, date, text = speech.get("id"), speech.get("speaker"), speech.get("party"), speech.get("date"), speech.get("text")
    normalized_date = normalize_date(date)
    content = format_header(speaker, party, normalized_date) + text
    metadata = {
        "source": file_path,
        "id": id,
        "speaker": speaker,
        "party": party,
        "date": normalized_date
    }
    return Document(page_content=content, metadata=metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load speeches from a JSONL file and create Document objects.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL file containing speeches.")
    args = parser.parse_args()

    documents, drop_count = load_speeches_from_jsonl(args.file_path)
    print(f"Loaded {len(documents)} documents from {args.file_path}, dropped {drop_count} entries without text.")
    print("=== Sample document content: ===")
    if documents:
        print(documents[0].page_content[:500])
        print("--- Sample document metadata: ----")
        print(documents[0].metadata)
