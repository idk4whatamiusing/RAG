"""SQL-backed corpus: dense via pgvector HNSW, sparse via tsvector GIN (GOA-11)."""
from __future__ import annotations
import socket
from typing import Callable

import boto3
import numpy as np
import psycopg

from src.retrieval import Hit, RRF_K, SELECTED_BOOST

DIM = 384
REGION = "ap-south-1"
DB = "voice-rag"


def rds_connection_factory(region: str = REGION, db: str = DB) -> Callable[[], psycopg.Connection]:
    def _connect() -> psycopg.Connection:
        ssm = boto3.client("ssm", region_name=region)
        pw = ssm.get_parameter(Name="/voice-rag/pg_password", WithDecryption=True)["Parameter"]["Value"]
        rds = boto3.client("rds", region_name=region)
        host = rds.describe_db_instances(DBInstanceIdentifier=db)["DBInstances"][0]["Endpoint"]["Address"]
        # ponytail: force IPv4 — NAT64 AAAA (64:ff9b::/96) stalls long streams from NATed hosts
        host = socket.getaddrinfo(host, 5432, socket.AF_INET)[0][4][0]
        conn = psycopg.connect(host=host, port=5432, user="postgres", password=pw,
                               dbname="postgres", connect_timeout=30, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 100")
        return conn
    return _connect


class SqlPart:
    """One language partition backed by the chunks table. Mirrors Index's retrieval API."""

    def __init__(self, lang: str, make_conn: Callable[[], psycopg.Connection]):
        self.lang = lang
        self._conn = make_conn()
        self._make_conn = make_conn

    def _cur(self):
        try:
            with self._conn.cursor() as probe:
                probe.execute("SELECT 1")
        except psycopg.OperationalError:
            self._conn = self._make_conn()
        return self._conn.cursor()

    @property
    def count(self) -> int:
        with self._cur() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE lang = %s", (self.lang,))
            return int(cur.fetchone()[0])

    def search_dense(self, query_vec: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        v = "[" + ",".join(f"{x:.6g}" for x in query_vec) + "]"
        with self._cur() as cur:
            cur.execute("SELECT doc, 1 - (vec <=> %s::vector) AS sim FROM chunks "
                        "WHERE lang = %s ORDER BY vec <=> %s::vector LIMIT %s",
                        (v, self.lang, v, k))
            return [(int(d), float(s)) for d, s in cur.fetchall()]

    def search_sparse(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        with self._cur() as cur:
            cur.execute("SELECT doc, ts_rank(to_tsvector('simple', text), "
                        "plainto_tsquery('simple', %s)) AS score FROM chunks "
                        "WHERE lang = %s AND to_tsvector('simple', text) @@ "
                        "plainto_tsquery('simple', %s) ORDER BY score DESC LIMIT %s",
                        (query, self.lang, query, k))
            return [(int(d), float(s)) for d, s in cur.fetchall()]

    def fetch(self, docs: list[int]) -> dict[int, dict]:
        if not docs:
            return {}
        ph = ",".join("%s" for _ in docs)
        with self._cur() as cur:
            cur.execute(f"SELECT doc, text, meta FROM chunks WHERE lang = %s AND doc IN ({ph})",
                        (self.lang, *docs))
            return {int(d): {"text": t, "meta": m} for d, t, m in cur.fetchall()}


class SqlCorpus:
    def __init__(self, make_conn: Callable[[], psycopg.Connection], langs: list[str] | None = None):
        self._make_conn = make_conn
        if langs is None:
            with make_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT lang FROM chunks ORDER BY 1")
                    langs = [r[0] for r in cur.fetchall()]
        self.parts: dict[str, SqlPart] = {l: SqlPart(l, make_conn) for l in langs}
        self.order = sorted(self.parts)

    def languages(self) -> list[str]:
        return list(self.order)

    def retrieve(self, query_vec: np.ndarray, query_text: str, lang: str, n: int = 20) -> list[Hit]:
        if lang not in self.parts:
            part = max(self.parts.values(), key=lambda p: p.count)
        else:
            part = self.parts[lang]
        dense = part.search_dense(query_vec, k=50)
        dense_sim = {doc: s for doc, s in dense}
        sparse = part.search_sparse(query_text, k=50)

        agg: dict[int, float] = {}
        for rk in (dense, sparse):
            for rank, (doc, _s) in enumerate(rk):
                agg[doc] = agg.get(doc, 0.0) + 1.0 / (RRF_K + rank)
        fused = sorted(agg.items(), key=lambda kv: -kv[1])[:n]

        rows = part.fetch([doc for doc, _ in fused])
        out = []
        for doc, score in fused:
            row = rows.get(doc)
            if row is None:
                continue
            meta = row["meta"]
            sel = int(meta.get("selected", 0)) == 1
            if sel:
                score += SELECTED_BOOST
            out.append(Hit(doc=doc, lang=part.lang, text=row["text"], meta=meta,
                           score=score, sim=dense_sim.get(doc, 0.0), selected=sel))
        return out