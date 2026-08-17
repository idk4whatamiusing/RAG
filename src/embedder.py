"""ONNX int8 multilingual embedder (384-dim, ~10-15ms CPU per query)."""
from __future__ import annotations
import os
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = os.environ.get("EMBED_MODEL", "Xenova/paraphrase-multilingual-MiniLM-L12-v2")
_TOKEN_FILE = "tokenizer.json"
_MODEL_FILE = "onnx/model_quantized.onnx"


class Embedder:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir
        from huggingface_hub import hf_hub_download
        tok_f = hf_hub_download(MODEL_ID, _TOKEN_FILE, cache_dir=cache_dir)
        onnx_f = hf_hub_download(MODEL_ID, _MODEL_FILE, cache_dir=cache_dir)
        self.tok = Tokenizer.from_file(tok_f)
        self.tok.enable_truncation(max_length=128)
        self.tok.enable_padding(pad_id=0)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort = ort.InferenceSession(onnx_f, sess_options=so, providers=["CPUExecutionProvider"])

    def encode(self, texts: list[str], batch: int = 16) -> np.ndarray:
        out = []
        tti = self.ort.get_inputs()[2].name if len(self.ort.get_inputs()) > 2 else None
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            att = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": att}
            if tti:
                feed[tti] = np.zeros_like(ids)
            emb = self.ort.run(None, feed)[0]
            mean = np.sum(emb * att[..., None], axis=1) / np.maximum(att.sum(1, keepdims=True), 1)
            out.append(mean)
        v = np.vstack(out)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_SCRIPT_RE = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]")


def langs_quick(text: str) -> bool:
    """Heuristic: does text contain Indic script characters (for fast routing)."""
    return bool(_SCRIPT_RE.search(text))