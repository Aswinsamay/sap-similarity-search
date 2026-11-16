"""Stage 08: how the ANN search of similarity/ann.py compares with exact search.

Run from the repo root, after `python -m similarity.ann build`:

    python -m evals.ann_benchmark measure            # recall and speed on the 30,000 real products
    python -m evals.ann_benchmark scale [--size N]   # 1,000,000 made-up vectors: minutes, about 2 GB of memory

measure  Step 1 alone (the 10 nearest product vectors, exact against HNSW at several efSearch)
         and both steps (the API's top 10, exact against shortlist + re-rank at several shortlist
         sizes), over 500 random products.
scale    Exact search against HNSW over made-up vectors, and today's exact search with all seven
         attributes over the same number of rows, to see how each grows with the catalogue.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from similarity.ann import CANDIDATES, nearest, new_index, product_index, rerank, shortlist
from similarity.features import Features
from similarity.search import SCORE_DECIMALS, WEIGHTS, load_features, rank, title_embeddings

K = 10                  # results compared with exact search
QUERIES = 500           # random products used to measure recall and speed
EF_VALUES = (10, 16, 32, 64, 128, 256)
CANDIDATE_VALUES = (100, 250, 500, 1000, 2000)
SEED = 0
TIE = 1e-5              # two cosines closer than this count as equal

# The scale test
SCALE_SIZE = 1_000_000
SCALE_QUERIES = 200
NOISE = 0.02            # how far a made-up vector is pushed from the real title it copies, per number
CHUNK = 100_000


# --- measuring against exact search ----------------------------------------------------------

def recall(found: np.ndarray, exact: np.ndarray, k: int = K) -> float:
    """Share of the k results found that score at least as well as the true k-th best.

    Counted by score, not by product: many products share a title, so several can tie with the
    k-th, and returning a different one of them is not a miss (as in ANN-Benchmarks).
    """
    found = np.pad(found[:k], (0, max(0, k - len(found))), constant_values=-np.inf)
    return float(np.mean(found >= exact[k - 1] - TIE))


def _ms(seconds: float, count: int) -> float:
    return seconds / count * 1000


def top_by_vector(vectors: np.ndarray, row: int, count: int) -> np.ndarray:
    """Exact search on vectors alone: the `count` rows closest to `row` (any order)."""
    cosine = vectors @ vectors[row]
    return np.argpartition(-cosine, count)[:count] if count < len(cosine) else np.arange(len(cosine))


def measure_vectors(index, vectors: np.ndarray, queries: np.ndarray, ef_values=EF_VALUES, k: int = K) -> pd.DataFrame:
    """Step 1 alone: the k nearest vectors by exact search and by HNSW at each efSearch.
    `vectors` are what the index holds; each query is searched on its own, as the API would."""
    exact, start = [], time.perf_counter()
    for query in queries:
        cosine = vectors @ query
        exact.append(np.sort(cosine[np.argpartition(-cosine, k)[:k]])[::-1])
    records = [{"search": "exact", "recall@10": 1.0, "ms per query": _ms(time.perf_counter() - start, len(queries))}]
    for ef in ef_values:
        found, start = [], time.perf_counter()
        for query in queries:
            found.append(nearest(index, query.reshape(1, -1), k, ef)[0][0])
        seconds = time.perf_counter() - start
        records.append({
            "search": f"HNSW, efSearch {ef}",
            "recall@10": np.mean([recall(f, e, k) for f, e in zip(found, exact)]),
            "ms per query": _ms(seconds, len(queries)),
        })
    return pd.DataFrame(records)


def measure_rerank(features: Features, rows, find: Callable[[int, int], np.ndarray],
                   candidate_values=CANDIDATE_VALUES, weights: dict[str, float] = WEIGHTS, k: int = K) -> pd.DataFrame:
    """Both steps: the final top k against exact search, for each shortlist size.
    `find(row, candidates)` returns the shortlist.

    recall@10   share of the top 10 that score at least as well as exact search's 10th
    same top 10 share of queries whose top 10 is exactly the same, in the same order
    """
    exact, start = {}, time.perf_counter()
    for row in rows:
        order, score = rank(features, row, weights)
        exact[row] = (order[:k], np.round(score, SCORE_DECIMALS))
    records = [{"search": "exact", "recall@10": 1.0, "same top 10": 1.0,
                "ms per query": _ms(time.perf_counter() - start, len(rows))}]
    for candidates in candidate_values:
        found, start = {}, time.perf_counter()
        for row in rows:
            found[row] = rerank(features, row, find(row, candidates), weights)
        seconds = time.perf_counter() - start
        records.append({
            "search": f"shortlist {candidates} + re-rank",
            "recall@10": np.mean([recall(np.round(found[r][1], SCORE_DECIMALS), exact[r][1], k) for r in rows]),
            "same top 10": np.mean([np.array_equal(found[r][0][:k], exact[r][0]) for r in rows]),
            "ms per query": _ms(seconds, len(rows)),
        })
    return pd.DataFrame(records)


def sample_rows(features: Features, count: int = QUERIES, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).choice(len(features.ids), size=count, replace=False)


# --- the scale test ------------------------------------------------------------------------

def made_up(base: np.ndarray, count: int, rng: np.random.Generator, noise: float = NOISE) -> np.ndarray:
    """`count` made-up vectors: copies of random real ones, nudged by noise, back to length 1.

    They cluster the way real products do. Purely random vectors would not: in hundreds of
    dimensions every pair of them is about equally far apart, which no catalogue looks like.
    """
    vectors = base[rng.integers(len(base), size=count)]
    vectors = vectors + noise * rng.standard_normal(vectors.shape, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def exact_full_ms(features: Features, size: int, queries: int = 20, seed: int = SEED) -> float:
    """ms per query of today's exact search (search.rank, all seven attributes) over a catalogue of
    `size` rows, made by repeating the real products. Every row is scored and sorted, as in the API."""
    plain = replace(features, title_embeddings=None, image_embeddings=None, ann_index=None)   # the API reads none
    big = plain.take(np.resize(np.arange(len(features.ids)), size))
    start = time.perf_counter()
    for row in np.random.default_rng(seed).choice(size, queries, replace=False):
        rank(big, int(row), WEIGHTS)
    return _ms(time.perf_counter() - start, queries)


def scale_test(features: Features, size: int = SCALE_SIZE, queries: int = SCALE_QUERIES,
               ef_values=EF_VALUES, seed: int = SEED, log=print) -> dict:
    """Exact search against HNSW over `size` made-up vectors.

    The made-up vectors copy the 384 title numbers, not the 1,214 of the product vector: a
    million of those would need 4.9 GB. The index is built with every thread, since only its
    time is reported. Exact search reads the index's own copy of the vectors, so they are held once.
    """
    import faiss

    base = title_embeddings(features).vectors
    rng = np.random.default_rng(seed)
    log(f"Exact search with all seven attributes over {size:,} rows (the real products repeated)")
    full_ms = exact_full_ms(features, size, seed=seed)

    index = new_index(base.shape[1])
    start = time.perf_counter()
    for done in range(0, size, CHUNK):
        index.add(made_up(base, min(CHUNK, size - done), rng))
        log(f"  {index.ntotal:,} / {size:,} vectors in the index, {time.perf_counter() - start:.0f} s")
    build_s = time.perf_counter() - start

    flat = faiss.downcast_index(index.storage)
    stored = faiss.rev_swig_ptr(flat.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
    query_vectors = made_up(base, queries, rng)
    log(f"Searching {queries} made-up queries")
    vectors = measure_vectors(index, stored, query_vectors, ef_values)

    start = time.perf_counter()
    for query in query_vectors:
        nearest(index, query.reshape(1, -1), CANDIDATES)
    shortlist_ms = _ms(time.perf_counter() - start, queries)
    return {"size": size, "build_s": build_s, "vectors": vectors,
            "shortlist_ms": shortlist_ms, "exact_full_ms": full_ms}


# --- command line ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["measure", "scale"])
    parser.add_argument("--size", type=int, default=SCALE_SIZE, help="made-up vectors in the scale test")
    args = parser.parse_args()
    features = load_features()
    pd.set_option("display.width", 120)

    if args.step == "measure":
        rows = sample_rows(features)
        index = product_index(features)
        vectors = index.reconstruct_n(0, index.ntotal)
        print(f"Step 1 alone, {len(rows)} random products: the 10 nearest product vectors")
        print(measure_vectors(index, vectors, vectors[rows]).round(3).to_string(index=False))
        print("\nBoth steps: the API's top 10 (the weights of method tfidf)")
        table = measure_rerank(features, rows, lambda row, n: shortlist(index, vectors[row], n))
        print(table.round(3).to_string(index=False))

    else:
        result = scale_test(features, args.size)
        print(f"\n{result['size']:,} made-up title vectors; index built in {result['build_s']:.0f} s")
        print(result["vectors"].round(3).to_string(index=False))
        print(f"Shortlist of {CANDIDATES}: {result['shortlist_ms']:.2f} ms per query")
        print(f"Exact search with all seven attributes: {result['exact_full_ms']:.0f} ms per query")


if __name__ == "__main__":
    main()
