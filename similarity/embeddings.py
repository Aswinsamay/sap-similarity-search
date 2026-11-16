"""Stage 06: one list of 384 numbers per product title, from a pretrained sentence transformer.

Run once from the repo root, after stage 01:

    python -m similarity.embeddings

Input:  data/processed/products.parquet (stage 01)
Output: data/processed/title_embeddings.npz (about 46 MB): the numbers for every title, the
        uniq_ids (to check the rows still line up with the products) and the score of a
        typical random pair (see `rescale`).

The model, all-MiniLM-L6-v2 (Reimers & Gurevych, 2019), reads the same product name as the
TF-IDF of stage 02. It was trained on about a billion pairs of sentences that mean the same
thing, so titles with the same meaning land close together even when they share no word
(frock / dress). Nothing is trained here: the model is used as downloaded.

The model is downloaded once from Hugging Face (about 90 MB) into the user's cache, not the
repo. Search only reads the saved file, so the API never needs the model or torch.

This file also holds what stage 07 (photos) shares with it: the file format, loading, and
the rescaling of scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from similarity import config
from similarity.preprocessing import load_products

# Settings: see similarity/config.py and .env.example
TITLE_MODEL = config.TITLE_MODEL
TITLE_PATH = config.TITLE_EMBEDDINGS_PATH
IMAGE_PATH = config.IMAGE_EMBEDDINGS_PATH   # written by stage 07
THREADS = config.ENCODE_THREADS             # CPU threads for encoding, kept low so the laptop does not run flat out

RANDOM_PAIRS = 10_000   # random pairs used to find the score that means "unrelated"
SEED = 0


@dataclass
class Embeddings:
    vectors: np.ndarray   # one row per product, each of length 1, so a dot product is the cosine
    present: np.ndarray   # False where a product has nothing to embed (no photo); all True for titles
    baseline: float       # median cosine of random pairs: the score of two unrelated products
    model: str

    def take(self, rows: np.ndarray) -> Embeddings:
        """Only these rows, in this order. The baseline stays that of the whole catalogue."""
        return replace(self, vectors=self.vectors[rows], present=self.present[rows])


# --- shared with stage 07 ----------------------------------------------------------------

def random_pair_baseline(vectors: np.ndarray, present: np.ndarray, pairs: int = RANDOM_PAIRS,
                         seed: int = SEED) -> float:
    """The median cosine of random pairs of different products.

    Embedding cosines are rarely near 0: two unrelated titles still share a language, and
    two unrelated photos are both clothes on a plain background. This number says where
    "unrelated" sits for one model.
    """
    rows = np.flatnonzero(present)
    rng = np.random.default_rng(seed)
    a, b = rng.choice(rows, pairs), rng.choice(rows, pairs)
    keep = a != b
    return float(np.median(np.einsum("ij,ij->i", vectors[a[keep]], vectors[b[keep]])))


def rescale(cosine: np.ndarray, baseline: float) -> np.ndarray:
    """Put an embedding cosine on the same footing as TF-IDF: unrelated 0, identical 1.

    (cosine - baseline) / (1 - baseline), and anything below a typical random pair is 0.
    Without this, every product would get a free score of about the baseline.
    """
    return np.clip((cosine - baseline) / (1 - baseline), 0.0, 1.0)


def save_embeddings(path: Path, ids: np.ndarray, vectors: np.ndarray, present: np.ndarray, model: str) -> Embeddings:
    """Write the numbers with the ids they belong to and the random-pair baseline."""
    vectors = vectors.astype(np.float32)
    baseline = random_pair_baseline(vectors, present)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids.astype(str), vectors=vectors, present=present, baseline=baseline, model=model)
    return Embeddings(vectors, present, baseline, model)


def load_embeddings(path: Path, ids: np.ndarray) -> Embeddings:
    """Read a saved file and check its rows are the same products, in the same order, as `ids`."""
    if not path.exists():
        command = "python -m similarity.embeddings" if path == TITLE_PATH else "python -m similarity.images embed"
        raise FileNotFoundError(f"{path} not found: run `{command}` first")
    with np.load(path) as data:
        if not np.array_equal(data["ids"], ids.astype(str)):
            raise ValueError(f"{path} was built for different products: run its stage again")
        return Embeddings(data["vectors"], data["present"], float(data["baseline"]), str(data["model"]))


# --- titles ------------------------------------------------------------------------------

def encode_titles(names: list[str], model_name: str = TITLE_MODEL, batch_size: int = 256) -> np.ndarray:
    """One row of numbers per title, each row of length 1."""
    import torch
    from sentence_transformers import SentenceTransformer   # imported here: only this step needs torch

    torch.set_num_threads(THREADS)
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(names, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def build_title_embeddings(products: pd.DataFrame, path: Path = TITLE_PATH) -> Embeddings:
    vectors = encode_titles(products["name"].tolist())
    return save_embeddings(path, products["uniq_id"].to_numpy(), vectors, np.ones(len(vectors), dtype=bool), TITLE_MODEL)


def main() -> None:
    products = load_products()
    start = time.perf_counter()
    embeddings = build_title_embeddings(products)
    seconds = time.perf_counter() - start
    print(f"Encoded {len(products):,} titles in {seconds:.0f} s with {embeddings.model}")
    print(f"Wrote {TITLE_PATH}: {embeddings.vectors.shape[0]:,} x {embeddings.vectors.shape[1]} numbers, "
          f"{TITLE_PATH.stat().st_size / 1e6:.0f} MB")
    print(f"A typical random pair of titles scores {embeddings.baseline:.2f}; rescaling maps that to 0")


if __name__ == "__main__":
    main()
