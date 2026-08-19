"""Load finished data/index/<lang>/ partitions into RDS pgvector (GOA-11)."""
from __future__ import annotations
import argparse
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import boto3
import faiss
import numpy as np
import pandas as pd
import psycopg

REGION = "ap-south-1"
DB = "voice-rag"
DIM = 384
BATCH = 5000


def _conn() -> psycopg.Connection:
    ssm = boto3.client("ssm", region_name=REGION)
    pw = ssm.get_parameter(Name="/voice-rag/pg_password", WithDecryption=True)["Parameter"]["Value"]
    rds = boto3.client("rds", region_name=REGION)
    host = rds.describe_db_instances(DBInstanceIdentifier=DB)["DBInstances"][0]["Endpoint"]["Address"]
    # ponytail: force IPv4 — NAT64 AAAA (64:ff9b::/96) stalls bulk COPY; ceiling: none, reconnect on instance replacement
    host = socket.getaddrinfo(host, 5432, socket.AF_INET)[0][4][0]
    return psycopg.connect(host=host, port=5432, user="postgres", password=pw, dbname="postgres",
                           connect_timeout=30, autocommit=True)


def _init(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
            id bigserial PRIMARY KEY,
            lang text NOT NULL,
            doc int NOT NULL,
            text text NOT NULL,
            meta jsonb NOT NULL,
            vec vector({DIM}) NOT NULL,
            UNIQUE (lang, doc))""")
    conn.commit()


def load_lang(conn: psycopg.Connection, path: Path, langs: list[str]):
    t0 = time.time()
    for lang in langs:
        d = path / lang
        if not (d / "faiss.index").exists():
            print(f"{lang}: missing index, skip")
            continue
        ix = faiss.read_index(str(d / "faiss.index"))
        n = ix.ntotal
        if not n:
            continue
        texts = pickle.loads((d / "texts.pkl").read_bytes())
        meta = pd.read_parquet(d / "meta.parquet").to_dict("records")
        assert len(texts) == n and len(meta) == n, f"{lang}: row mismatch {len(texts)}/{n}/{len(meta)}"
        vecs = ix.reconstruct_n(0, n).astype(np.float32)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE lang = %s", (lang,))
        with conn.cursor() as cur:
            with cur.copy("COPY chunks (lang, doc, text, meta, vec) FROM STDIN") as cp:
                for i in range(n):
                    cp.write_row((lang, i, texts[i], json.dumps(meta[i], ensure_ascii=False),
                                  "[" + ",".join(f"{v:.6g}" for v in vecs[i]) + "]"))
                    if i and i % 25000 == 0:
                        print(f"  {lang}: {i}/{n} rows, {time.time()-t0:.0f}s", flush=True)
        print(f"{lang}: {n} rows loaded, {time.time()-t0:.0f}s")
        with conn.cursor() as cur:
            cur.execute(f"CREATE INDEX IF NOT EXISTS chunks_vec_{lang} ON chunks "
                        f"USING hnsw (vec vector_cosine_ops) WHERE lang = '{lang}'")
            cur.execute(f"CREATE INDEX IF NOT EXISTS chunks_tsv_{lang} ON chunks "
                        f"USING gin (to_tsvector('simple', text)) WHERE lang = '{lang}'")
        print(f"{lang}: indexes created, {time.time()-t0:.0f}s")
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--langs", nargs="*", default=None, help="default: all dirs present")
    args = ap.parse_args()
    base = Path(args.index_dir)
    langs = args.langs or [p.name for p in sorted(base.iterdir()) if p.is_dir()]
    conn = _conn()
    _init(conn)
    load_lang(conn, base, langs)
    conn.close()
