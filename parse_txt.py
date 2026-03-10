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

    # Split into category sections separated by lines of underscores or hyphens
    sections = re.split(r"_{5,}|-{5,}", text)

    knowledge_base: list[dict] = []
    current_category = "General"

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.splitlines()

        # Check if section has Q&A format
        joined = "\n".join(lines)
        qa_pairs = re.findall(
            r"Q\d+\.\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\nQ\d+\.|\Z)",
            joined,
            re.DOTALL,
        )

        if qa_pairs:
            # It's a structured Q&A section
            header_line = ""
            for line in lines:
                stripped = line.strip()
                cleaned = re.sub(r"^[^\w]*", "", stripped)
                if cleaned and not re.match(r"^Q\d+\.", cleaned) and not re.match(r"^A:", cleaned):
                    header_line = cleaned
                    break

            if header_line:
                current_category = header_line

            for question, answer in qa_pairs:
                knowledge_base.append(
                    {
                        "category": current_category,
                        "question": question.strip(),
                        "answer": answer.strip(),
                    }
                )
        else:
            # It's an unstructured section (e.g., Code of Conduct)
            current_heading = "Institutional Document"
            current_prefix = ""
            current_buffer = []

            def flush_buffer(is_bullet=False):
                nonlocal current_prefix
                if not current_buffer:
                    return
                para = " ".join(current_buffer).strip()
                # Remove strange unicode bullets and normal bullets
                para = re.sub(r'[\uf0b7\uf02d\uf0a7\uf0d8\uf0fc\uf0e0\xad\x81]', '', para)
                para = re.sub(r'^[o\-\*]\s*', '', para).strip()
                
                if len(para.split()) > 3:
                    ans = para
                    # Append prefix if it exists and paragraph doesn't already start with it
                    if current_prefix and not para.startswith(current_prefix):
                        ans = f"{current_prefix} {para}"
                    knowledge_base.append({
                        "category": "Unstructured Knowledge",
                        "question": f"Topic: {current_heading}",
                        "answer": ans
                    })
                    
                    if para.endswith(":"):
                        current_prefix = para
                    elif not is_bullet:
                        current_prefix = ""
                current_buffer.clear()

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    flush_buffer()
                    continue
                    
                # Ignore page numbers and TOC entries
                if stripped.isdigit() or stripped.lower() in ("contents", "page no", "introduction"):
                    continue
                    
                # Detect headings: short lines, no ending punctuation
                has_letters = bool(re.search(r'[A-Za-z]', stripped))
                is_bullet = stripped.startswith(("", "", "o ", "", "", "-", "*"))
                
                if len(stripped.split()) <= 12 and not stripped.endswith((".", ":", ";", ",", "!", "?")) and has_letters and not is_bullet:
                    flush_buffer()
                    current_heading = stripped
                    current_prefix = ""  # Reset prefix on new heading
                else:
                    if is_bullet:
                         flush_buffer()
                         current_buffer.append(stripped)
                         flush_buffer(is_bullet=True)
                    else:
                         current_buffer.append(stripped)
                         
            flush_buffer()

    return knowledge_base


def main():
    kb = parse_txt_to_json(TXT_FILE)
    JSON_FILE.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Parsed {len(kb)} Q&A pairs → {JSON_FILE.name}")


if __name__ == "__main__":
    main()
