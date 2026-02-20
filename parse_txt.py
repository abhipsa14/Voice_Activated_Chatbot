"""
Parses the UIT Q&A text file into a structured JSON file.
Run this once to generate knowledge_base.json from uit.txt.
"""

import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TXT_FILE = BASE_DIR / "uit.txt"
JSON_FILE = BASE_DIR / "knowledge_base.json"


def parse_txt_to_json(txt_path: Path) -> list[dict]:
    """Read the Q&A text file and return a list of {category, question, answer} dicts."""
    text = txt_path.read_text(encoding="utf-8")

    # Split into category sections separated by lines of underscores
    sections = re.split(r"_{5,}", text)

    knowledge_base: list[dict] = []
    current_category = "General"

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.splitlines()

        # Detect category header (first non-empty line that is NOT a Q/A)
        header_line = ""
        for line in lines:
            stripped = line.strip()
            # Skip emoji prefixes
            cleaned = re.sub(r"^[^\w]*", "", stripped)
            if cleaned and not re.match(r"^Q\d+\.", cleaned) and not re.match(r"^A:", cleaned):
                header_line = cleaned
                break

        if header_line:
            current_category = header_line

        # Extract Q&A pairs
        joined = "\n".join(lines)
        qa_pairs = re.findall(
            r"Q\d+\.\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\nQ\d+\.|\Z)",
            joined,
            re.DOTALL,
        )

        for question, answer in qa_pairs:
            knowledge_base.append(
                {
                    "category": current_category,
                    "question": question.strip(),
                    "answer": answer.strip(),
                }
            )

    return knowledge_base


def main():
    kb = parse_txt_to_json(TXT_FILE)
    JSON_FILE.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Parsed {len(kb)} Q&A pairs → {JSON_FILE.name}")


if __name__ == "__main__":
    main()
