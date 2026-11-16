"""Stage 08: the product vector, the HNSW index, the two-step search and its measurements, on a
small made-up catalogue. No saved file is needed; skipped when faiss-cpu is not installed."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from evals import ann_benchmark as bench  # noqa: E402
from similarity import ann  # noqa: E402
from similarity.embeddings import Embeddings  # noqa: E402
from similarity.features import build_features  # noqa: E402
from similarity.search import WEIGHTS, attribute_scores, rank, search  # noqa: E402

KINDS = ["Frock", "Saree", "Kurta", "Shirt", "Cap"]
GENDERS = ["Girls", "Womens", "Mens", "Boys"]


def catalogue(size=80, seed=0):
    """Products that differ in kind, gender, brand and price, with some of each missing,
    and made-up title numbers that cluster by kind."""
    rows = []
    for i in range(size):
        kind, gender = KINDS[i % 5], GENDERS[i % 4]
        rows.append({
            "uniq_id": f"p{i:03d}", "asin": f"A{i}", "name": f"{gender} {kind} style {i % 3}",
            "rating_adjusted": 3 + (i % 5) / 2, "brand": None if i % 9 == 0 else f"B{i % 7}", "colours": None,
            "price": np.nan if i % 11 == 0 else 100.0 * (1 + i % 13), "weight_g": np.nan,
            "child_category": f"{gender}{kind}", "browsenode": str(i % 5),
        })
    features = build_features(pd.DataFrame(rows))
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((len(KINDS), 16))
    vectors = centres[np.arange(size) % 5] + 0.3 * rng.standard_normal((size, 16))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    features.title_embeddings = Embeddings(vectors, np.ones(size, dtype=bool), 0.0, "toy")
    features.ann_index = ann.build_index(ann.product_vectors(features))
    return features


def test_take_keeps_every_pair_score():
    features = catalogue()
    rows = np.array([5, 0, 17, 42])
    part = features.take(rows)
    full = attribute_scores(features, 5, WEIGHTS)
    for name, (score, present) in attribute_scores(part, 0, WEIGHTS).items():
        assert score == pytest.approx(full[name][0][rows]) and (present == full[name][1][rows]).all()


def test_a_shortlist_of_everything_is_exact_search():
    features = catalogue()
    for row in [0, 7, 33]:
        exact_order, exact_score = rank(features, row)
        order, score = ann.ann_rank(features, row, candidates=len(features.ids))
        assert order.tolist() == exact_order.tolist() and score == pytest.approx(exact_score)


def test_search_mode_ann_uses_the_index():
    features = catalogue()
    exact_order, _ = search(features, 7, "tfidf", "exact")
    ann_order, _ = search(features, 7, "tfidf", "ann")
    assert ann_order.tolist() == ann.ann_rank(features, 7)[0].tolist()
    assert ann_order.tolist() == exact_order.tolist()   # 80 products, a shortlist of 1,000: all are scored


def test_product_vectors_follow_the_attributes():
    features = catalogue()
    vectors = ann.product_vectors(features)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)
    # 1 and 21 are both women's sarees (brand and price differ); 1 and 2 share nothing
    assert vectors[1] @ vectors[21] > vectors[1] @ vectors[2]


def test_twice_the_price_shares_half_the_bands():
    bands = ann.price_bands(np.array([100.0, 200.0, 400.0, np.nan]))
    assert bands[0] @ bands[1] == pytest.approx(0.5, abs=0.02)     # like smaller ÷ larger = 0.5
    assert bands[0] @ bands[2] < 0.1
    assert not bands[3].any()                                      # missing: all 0, so no match


def test_recall_counts_a_tie_as_found():
    exact = np.array([0.9, 0.8, 0.7])
    assert bench.recall(np.array([0.9, 0.8, 0.7]), exact, k=3) == 1.0
    assert bench.recall(np.array([0.9, 0.7, 0.7]), exact, k=3) == 1.0     # a different product tied with the 3rd
    assert bench.recall(np.array([0.9, 0.8, 0.6]), exact, k=3) == pytest.approx(2 / 3)
    assert bench.recall(np.array([0.9]), exact, k=3) == pytest.approx(1 / 3)   # too few found


def test_hnsw_finds_nearly_all_nearest_vectors():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((50, 32)).astype(np.float32)
    vectors = bench.made_up(base / np.linalg.norm(base, axis=1, keepdims=True), 3000, rng, noise=0.1)
    index = ann.build_index(vectors)
    table = bench.measure_vectors(index, vectors, vectors[:100], ef_values=(64,))
    assert table.loc[1, "recall@10"] >= 0.95


def test_the_saved_index_checks_its_vectors(tmp_path):
    vectors = ann.product_vectors(catalogue())
    path = tmp_path / "index.faiss"
    ann.save_index(ann.build_index(vectors), path)
    assert ann.load_index(path, vectors).ntotal == len(vectors)
    changed = vectors.copy()
    changed[3] = changed[4]
    with pytest.raises(ValueError, match="build"):
        ann.load_index(path, changed)                              # the products changed since the build
    with pytest.raises(FileNotFoundError, match="python -m similarity.ann build"):
        ann.load_index(tmp_path / "missing.faiss", vectors)


def test_the_api_never_imports_faiss():
    code = "import sys, app, evals.run_eval; print('faiss' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
