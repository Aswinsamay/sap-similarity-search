"""Stage 06: the shared embedding helpers and the minilm method, with small made-up numbers.
No model is downloaded and no saved file is needed."""

import numpy as np
import pandas as pd
import pytest

import similarity.search as search
from similarity.embeddings import Embeddings, load_embeddings, random_pair_baseline, rescale, save_embeddings
from similarity.features import build_features
from similarity.search import METHOD_WEIGHTS, WEIGHTS, attribute_scores, embedding_scores, method_weights, rank


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_rescale_puts_random_at_zero_and_identical_at_one():
    scores = rescale(np.array([0.6, 1.0, 0.3, 0.8]), baseline=0.6)
    assert scores == pytest.approx([0.0, 1.0, 0.0, 0.5])      # below a random pair is 0, not negative


def test_baseline_ignores_products_without_embeddings():
    vectors = np.array([unit(1, 0), unit(1, 0), unit(1, 0), unit(0, 1)])
    present = np.array([True, True, True, False])              # the odd one out has no photo
    assert random_pair_baseline(vectors, present, pairs=200) == pytest.approx(1.0)


def test_saved_embeddings_check_their_rows(tmp_path):
    path = tmp_path / "toy.npz"
    ids = np.array(["a", "b", "c"])
    vectors = np.array([unit(1, 0), unit(0, 1), unit(1, 1)])
    saved = save_embeddings(path, ids, vectors, np.ones(3, dtype=bool), "toy")
    loaded = load_embeddings(path, ids)
    assert loaded.vectors == pytest.approx(vectors) and loaded.baseline == pytest.approx(saved.baseline)
    assert loaded.model == "toy"
    with pytest.raises(ValueError):
        load_embeddings(path, ids[::-1])                       # same products in another order
    with pytest.raises(FileNotFoundError, match="python -m"):
        load_embeddings(tmp_path / "missing.npz", ids)


def test_minilm_swaps_only_the_title_score():
    minilm = method_weights("minilm")
    assert METHOD_WEIGHTS["tfidf"] == WEIGHTS
    assert "text" not in minilm and minilm["text_embedding"] == WEIGHTS["text"]
    assert {k: v for k, v in minilm.items() if k != "text_embedding"} == {k: v for k, v in WEIGHTS.items() if k != "text"}
    with pytest.raises(ValueError):
        method_weights("word2vec")


def toy_features():
    base = {"brand": None, "colours": None, "price": 500.0, "weight_g": np.nan,
            "child_category": "GirlsDressesJumpsuits", "browsenode": "1"}
    rows = [
        {**base, "uniq_id": "q", "asin": "A1", "name": "Girls Frock", "rating_adjusted": 4.0},   # 0 the query
        {**base, "uniq_id": "a", "asin": "A2", "name": "Kids Dress", "rating_adjusted": 3.0},    # 1 same meaning
        {**base, "uniq_id": "b", "asin": "A3", "name": "Silk Saree", "rating_adjusted": 4.9},    # 2 different
        {**base, "uniq_id": "c", "asin": "A4", "name": "Silk Saree", "rating_adjusted": 4.9},    # 3 gives the words a vector
    ]
    return build_features(pd.DataFrame(rows))


def test_embedding_scores_skip_products_without_embeddings():
    embeddings = Embeddings(np.array([unit(1, 0), unit(1, 0), unit(0, 1)]), np.array([True, False, True]), 0.0, "toy")
    score, present = embedding_scores(embeddings, 0)
    assert present.tolist() == [True, False, True]
    assert score[2] == pytest.approx(0.0)


def test_minilm_finds_a_synonym_tfidf_cannot():
    features = toy_features()
    # made-up title numbers: frock and dress point the same way, saree elsewhere
    vectors = np.array([unit(1, 0.1), unit(1, 0.2), unit(0, 1), unit(0, 1)])
    features.title_embeddings = Embeddings(vectors, np.ones(4, dtype=bool), 0.0, "toy")
    tfidf_first = features.ids[rank(features, 0, METHOD_WEIGHTS["tfidf"])[0][0]]
    minilm_first = features.ids[rank(features, 0, METHOD_WEIGHTS["minilm"])[0][0]]
    assert tfidf_first == "b"      # no shared words anywhere: the rating decides
    assert minilm_first == "a"     # the dress is closest in meaning


def test_api_weights_never_read_the_embedding_file(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "TITLE_PATH", tmp_path / "missing.npz")
    features = toy_features()
    assert "text_embedding" not in attribute_scores(features, 0, WEIGHTS)
    rank(features, 0, WEIGHTS)                                 # works without the file
    with pytest.raises(FileNotFoundError):
        rank(features, 0, METHOD_WEIGHTS["minilm"])            # minilm needs it, and says so
