"""Stream per-language MSMARCO-XI parquet row groups (nested-column safe).

fetch_parquet.RangeFile serves pyarrow with the row-group bytes + footer as one
contiguous block, so nested list<struct> columns build without chunking.
"""
from __future__ import annotations
import io, time
import aiohttp, fsspec, pyarrow.parquet as pq

DATA_ROOT = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train"


def connect() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem(
        "http",
        client_kwargs={"timeout": aiohttp.ClientTimeout(total=300, sock_read=240)},
    )


class RangeFile:
    """File-like over disjoint cached byte blocks, offset-aware (pyarrow-safe)."""

    def __init__(self, blocks: list[tuple[int, bytes]]):
        self.blocks = sorted(blocks)  # (start, data) non-overlapping
        self.size = max(s + len(d) for s, d in self.blocks)
        self.pos = 0
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        out, want = [], self.pos + (n if n >= 0 else self.size)
        for start, data in self.blocks:
            if start + len(data) <= self.pos:
                continue
            if start >= want:
                break
            a, b = max(self.pos, start), min(want, start + len(data))
            if a < b:
                out.append(data[a - start : b - start])
                self.pos = b
        return b"".join(out)

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def close(self) -> None:
        self.closed = True


def iter_row_groups(lang_file: str, fs: fsspec.AbstractFileSystem | None = None,
                    use_cache: bool = True):
    """Yield row groups of train/{lang_file} as pyarrow Tables (single chunk)."""
    url = f"{DATA_ROOT}/{lang_file}"
    fs = fs or connect()
    size = fs.size(url)
    tail = 1 << 16
    blocks: list[tuple[int, bytes]] = []

    def set_tail() -> None:
        blocks[:] = [(size - tail, fetch(size - tail, size))]

    def fetch(start: int, end: int) -> bytes:
        with fs.open(url, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def footer_parquet():
        if not blocks:
            set_tail()
        f = io.BytesIO(blocks[0][1])
        try:
            return pq.ParquetFile(f)
        except Exception:
            nonlocal tail
            tail *= 2
            set_tail()
            return pq.ParquetFile(io.BytesIO(blocks[0][1]))

    pf = footer_parquet()
    true_eof = blocks[0][0] + len(blocks[0][1])  # clamped fetch ends at EOF
    for rg in range(pf.metadata.num_row_groups):
        meta = pf.metadata.row_group(rg)
        start = meta.column(0).file_offset
        alen = sum(meta.column(c).total_compressed_size for c in range(meta.num_columns))
        spine = sorted((meta.column(c).file_offset, c) for c in range(meta.num_columns))
        end = max(o + meta.column(c).total_compressed_size for o, c in spine)
        end = min(end, true_eof)
        chunk = fetch(start, end)  # one contiguous block -> single chunk
        with pq.ParquetFile(RangeFile([(0, chunk), blocks[0]])) as pg:
            yield pg.read_row_group(rg)


if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else "hintrain.parquet"
    t0 = time.perf_counter()
    fs = connect()
    for i, tab in enumerate(iter_row_groups(lang, fs)):
        if i >= 2:
            break
        qs = tab.column("query").combine_chunks().to_pylist()
        ps = tab.column("passages").combine_chunks().to_pylist()
        print(f"rg{i}: {tab.num_rows} rows, sel {sum(1 for s in ps[0]['is_selected'] if s==1)}/{len(ps[0]['is_selected'])}, q={qs[0][:50]!r}")
    print(f"total {time.perf_counter()-t0:.1f}s")