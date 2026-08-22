"""GOA-17 G1/H2/H3: build the curated knowledge partitions (goa + docs) and save index artifacts."""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_passage
from src.embedder import Embedder
from src.indexer import Index


def load_txt(path: Path, prefix: str, start: int) -> tuple[list[dict], int]:
    chunks = []
    n = start
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t:
            continue
        n += 1
        chunks.extend(chunk_passage(str(n), t, "native", lang=path.parent.stem,
                                    qid=f"{prefix}:{n}", selected=1, qtype=""))
    return chunks, n


def load_md(paths: list[Path], prefix: str, start: int) -> tuple[list[dict], int]:
    chunks = []
    n = start
    for path in paths:
        txt = path.read_text(encoding="utf-8")
        parts = [p.strip() for p in re.split(r"\n{2,}|(?=^## )", txt, flags=re.M) if p.strip()]
        for p in parts:
            p = re.sub(r"^#+\s*", "", p).strip()
            if len(p) < 40:
                continue
            n += 1
            chunks.extend(chunk_passage(str(n), p[:700], "native", lang="docs",
                                        qid=f"{prefix}:{n}", selected=1, qtype=""))
            chunks = chunks[-400:]  # guard: cap docs partition size
    return chunks, n


def build(lang: str, chunks: list[dict], out: Path):
    embedder = Embedder()
    ix = Index(lang, embedder)
    for i in range(0, len(chunks), 128):
        b = chunks[i:i + 128]
        ix.add([c["text"] for c in b], [c["meta"] for c in b],
               embedder.encode([c["text"] for c in b]), index_toks=False)
    ix.save(out / lang)
    print(f"{lang}: {len(chunks)} chunks -> {out / lang} ({ix.count})")


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "index"
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    all_chunks = []
    for f, pref in (("data/goa_hi.txt", "goh"), ("data/goa_en.txt", "goe"), ("data/goa_kok.txt", "gok")):
        chunks, n = load_txt(root / f, pref, n)
        all_chunks.extend(chunks)
    build("goa", all_chunks, out)

    doc_chunks, _ = load_md([root / "docs" / "bench-results.md", root / "docs" / "requirements.md"],
                            "doc", 0)
    build("docs", doc_chunks, out)
    print("done")


if __name__ == "__main__":
    main()
