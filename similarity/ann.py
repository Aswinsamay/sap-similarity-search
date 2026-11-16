"""Stage 08 (Part 3): approximate nearest neighbour (ANN) search with an HNSW index.

Run from the repo root, after stage 06 (`python -m similarity.embeddings`):

    python -m similarity.ann build     # the index: data/processed/product_hnsw.faiss, about a minute

How it compares with exact search is measured in evals/ann_benchmark.py.

Exact search scores the query against every product, so its time grows with the catalogue.
HNSW, Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016; IEEE TPAMI 2018),
links every vector to a few of its nearest neighbours, in layers: a sparse top layer for long
jumps, down to a bottom layer that holds every vector. A search starts at the top, walks
towards the query, and drops a layer when it can get no closer. It looks at a few thousand
vectors instead of all of them, and can miss a true neighbour: that is the "approximate".
The index comes from FAISS (Johnson, Douze & Jégou, 2017).

An index needs one list of numbers per product, and our score is a weighted mix of seven
attributes. So the search has two steps:
    1. shortlist: the index returns the `candidates` products whose product vectors (below)
       are closest to the query's;
    2. re-rank: only those are scored with the usual weighted score (stage 03) and sorted.
Re-ranking is exact, so a result can only differ from exact search when a product that
belongs in the top 10 was left off the shortlist.

The API uses exact search unless the setting SEARCH_MODE is ann (see .env.example). faiss-cpu is in
requirements-dev.txt only, and this file imports it only inside its functions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from similarity import config
from similarity.features import Features
from similarity.search import WEIGHTS, image_embeddings, load_features, rank, title_embeddings

INDEX_PATH = config.ANN_INDEX_PATH   # settings: see similarity/config.py and .env.example

# The product vector
BRAND_SLOTS = 256                       # brands are hashed into this many slots
PRICE_RANGE = (10.0, 200_000.0)         # rupees covered by the price bands; prices outside are clipped
PRICE_WIDTH = np.sqrt(np.log(2) / 4)    # band width on a log scale: twice the price shares half the bands

# The index
# At M 32 and EF_CONSTRUCTION 200, 41 products (0.14%) could not be reached at all: a search with
# their own vector did not find them, even 1,000 deep. At 64 and 500 every product is found.
M = config.ANN_M                       # links per vector in each layer (the paper's M): more links, fewer misses, bigger index
EF_CONSTRUCTION = config.ANN_EF_CONSTRUCTION   # how many candidates are weighed while linking each new vector into the graph
CANDIDATES = config.ANN_CANDIDATES     # shortlist size: the products re-scored with the full weighted score
# How many candidates a search keeps in view. A search keeps at least as many as it returns, so the
# shortlist of CANDIDATES keeps that many; this only matters for smaller searches (evals/ann_benchmark.py).
EF_SEARCH = 64


# --- the product vector ------------------------------------------------------------------------

def one_hot(codes: np.ndarray, size: int) -> np.ndarray:
    """A 1 in each product's slot; a row of 0s where the code is -1 (missing)."""
    out = np.zeros((len(codes), size), dtype=np.float32)
    known = np.flatnonzero(codes >= 0)
    out[known, codes[known]] = 1
    return out


def price_bands(price: np.ndarray) -> np.ndarray:
    """Each price as overlapping bands on a log scale, scaled to length 1; 0s where it is missing.
    Two prices a factor x apart get a dot product of about exp(-ln(x)² / 4w²) for band width w:
    0.5 for twice the price, the same as the API's smaller ÷ larger."""
    centres = np.arange(np.log(PRICE_RANGE[0]), np.log(PRICE_RANGE[1]) + PRICE_WIDTH, PRICE_WIDTH / 2)
    log_price = np.log(np.clip(price, *PRICE_RANGE))
    bands = np.exp(-((log_price[:, None] - centres) ** 2) / (2 * PRICE_WIDTH**2))
    return np.nan_to_num(bands / np.linalg.norm(bands, axis=1, keepdims=True)).astype(np.float32)


def product_vectors(features: Features) -> np.ndarray:
    """One list of numbers per product whose dot products follow the API's score: 1,214 here
    (384 title + 520 category and browsenode + 4 gender + 256 brand + 50 price).

    Each attribute becomes a block of numbers of length 1, so the dot product of two blocks is
    that attribute's similarity:
        title     the MiniLM numbers of stage 06, standing in for TF-IDF (7,412 mostly empty numbers)
        category  a 1 in the slot of the child category, or of the browsenode when there is none
        gender    a 1 in the slot of the gender
        brand     a 1 in one of 256 slots (the brand's code, hashed): two brands share one 1 time in 256
        price     overlapping bands (price_bands)
    Each block is multiplied by the square root of its weight, so the dot product of two whole
    vectors is the weighted sum. Colour and weight are left out: 5% each, missing for 80% of
    products. A missing attribute is a block of 0s; scaling every vector to length 1 then lets
    its other blocks count for more, much like the API skipping what is missing.
    """
    first_node = features.category.max() + 1
    category = np.where(features.category >= 0, features.category,
                        np.where(features.browsenode >= 0, first_node + features.browsenode, -1))
    brand = np.where(features.brand >= 0, features.brand % BRAND_SLOTS, -1)
    blocks = {
        "text": title_embeddings(features).vectors,
        "category": one_hot(category, first_node + features.browsenode.max() + 1),
        "gender": one_hot(features.gender, features.gender.max() + 1),
        "brand": one_hot(brand, BRAND_SLOTS),
        "price": price_bands(features.price),
    }
    vectors = np.hstack([np.sqrt(WEIGHTS[name]) * block for name, block in blocks.items()]).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --- the index ------------------------------------------------------------------------------

def new_index(dimensions: int):
    """An empty HNSW index. Inner product is the cosine here, since every vector has length 1."""
    import faiss

    index = faiss.IndexHNSWFlat(dimensions, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    return index


def build_index(vectors: np.ndarray):
    """An HNSW index of the vectors: row i of the index is row i of `vectors`.
    Built with one thread, so the same vectors always give the same graph and the same results."""
    import faiss

    index = new_index(vectors.shape[1])
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    finally:
        faiss.omp_set_num_threads(threads)
    return index


def save_index(index, path: Path = INDEX_PATH) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path, vectors: np.ndarray):
    """Read a saved index and check it holds exactly these vectors, row for row, so it cannot
    silently fall out of step with the products after they are rebuilt."""
    import faiss

    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m similarity.ann build` first")
    index = faiss.read_index(str(path))
    if index.ntotal != len(vectors) or not np.array_equal(index.reconstruct_n(0, index.ntotal), vectors):
        raise ValueError(f"{path} does not match the products: run `python -m similarity.ann build` again")
    return index


def product_index(features: Features):
    """The index of the product vectors: read on first use, then kept with the features."""
    if features.ann_index is None:
        features.ann_index = load_index(INDEX_PATH, product_vectors(features))
    return features.ann_index


# --- the two-step search --------------------------------------------------------------------

def nearest(index, vectors: np.ndarray, k: int, ef_search: int = EF_SEARCH) -> tuple[np.ndarray, np.ndarray]:
    """(cosines, rows) of the k vectors HNSW finds closest to each row of `vectors`, closest first.
    A row of -1 marks an empty slot when the index holds fewer than k."""
    import faiss

    params = faiss.SearchParametersHNSW(efSearch=max(ef_search, k))
    return index.search(np.ascontiguousarray(vectors, dtype=np.float32), k, params=params)


def shortlist(index, vector: np.ndarray, candidates: int = CANDIDATES) -> np.ndarray:
    """Step 1: the rows of the `candidates` products whose vectors are closest to `vector`."""
    _, rows = nearest(index, vector.reshape(1, -1), candidates)
    return rows[0][rows[0] >= 0]


def rerank(features: Features, row: int, candidates: np.ndarray, weights: dict[str, float] = WEIGHTS):
    """Step 2: search.rank over the candidates only. Rows are those of the full table."""
    rows = np.union1d(candidates, [row])
    order, score = rank(features.take(rows), int(np.searchsorted(rows, row)), weights)
    return rows[order], score


def ann_rank(features: Features, row: int, weights: dict[str, float] = WEIGHTS,
             candidates: int = CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """Like search.rank, but only the shortlist is scored."""
    if "image" in weights:
        image_embeddings(features)   # read before the cut, so the shortlist carries the photos
    index = product_index(features)  # also reads the title embeddings
    return rerank(features, row, shortlist(index, index.reconstruct(int(row)), candidates), weights)


# --- command line ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build"])
    parser.parse_args()
    start = time.perf_counter()
    index = build_index(product_vectors(load_features()))
    save_index(index)
    size = INDEX_PATH.stat().st_size / 1e6
    print(f"Wrote {INDEX_PATH}: {index.ntotal:,} products x {index.d:,} numbers, {size:.0f} MB, "
          f"in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
