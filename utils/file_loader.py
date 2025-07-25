from pathlib import Path
from pdfminer.high_level import extract_text
from docx import Document

def load_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return Path(filepath).read_text(encoding="utf-8")
    elif ext == ".pdf":
        return extract_text(filepath)
    elif ext == ".docx":
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")
