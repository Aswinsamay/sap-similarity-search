import numpy as np
import pandas as pd
import pytest

from similarity.features import build_features
from similarity.preprocessing import PROCESSED_PATH
from similarity.search import (
    WEIGHTS,
    attribute_scores,
    category_scores,
    code_scores,
    colour_scores,
    combine,
    find_similar_products,
    rank,
    ratio_scores,
    score_breakdown,
    text_scores,
)


def small_features():
    shirt = {
        "name": "Men's Cotton T-Shirt", "brand": "max", "colours": ["black"], "price": 200.0,
        "weight_g": np.nan, "child_category": "MensT_Shirts", "browsenode": "1",
    }
    kurta = {
        "name": "Women's Silk Kurta", "brand": None, "colours": None, "price": 900.0,
        "weight_g": np.nan, "child_category": "WomensKurtasKurtis", "browsenode": "2",
    }
    rows = [
        {**shirt, "uniq_id": "q", "asin": "A1", "rating_adjusted": 4.0},     # 0 the query
        {**shirt, "uniq_id": "q2", "asin": "A1", "rating_adjusted": 4.0},    # 1 the query listed twice
        {**shirt, "uniq_id": "s1", "asin": "A2", "rating_adjusted": 3.0},    # 2 identical to the query
        {**shirt, "uniq_id": "s2", "asin": "A3", "rating_adjusted": 4.5},    # 3 identical, better rating
        {**shirt, "uniq_id": "s1b", "asin": "A2", "rating_adjusted": 3.0},   # 4 s1 listed twice
        {**kurta, "uniq_id": "d", "asin": "A4", "rating_adjusted": 4.9},     # 5 different
        {**kurta, "uniq_id": "b", "asin": "A5", "rating_adjusted": 4.0,      # 6 no category, query's browsenode
         "child_category": None, "browsenode": "1", "price": np.nan},
        {**kurta, "uniq_id": "e", "asin": "A6", "rating_adjusted": 4.0,      # 7 no word shared with any title
         "name": "Bauer Uncategorized", "child_category": None, "browsenode": "1", "price": np.nan},
    ]
    return build_features(pd.DataFrame(rows))


def test_ratio_scores():
    score, present = ratio_scores(np.array([100.0, 200.0, np.nan]), 0)
    assert score.tolist() == [1.0, 0.5, 0.0]
    assert present.tolist() == [True, True, False]


def test_two_missing_codes_are_not_a_match():
    codes = np.array([-1, -1, 0])
    assert code_scores(codes, 0)[1].tolist() == [False, False, False]
    assert code_scores(codes, 2)[1].tolist() == [False, False, True]


def test_category_falls_back_to_browsenode():
    score, present = category_scores(small_features(), 0)
    assert present[6] and score[6] == 1.0         # no category, same browsenode
    assert present[5] and score[5] == 0.0         # both have a category, and it differs


def test_title_without_shared_words_scores_zero():
    features = small_features()
    assert features.text[7].nnz == 0
    score, present = text_scores(features, 0)
    assert present[7] and score[7] == 0.0
    # judged on text (0) and browsenode (1), not on the browsenode alone
    assert combine(attribute_scores(features, 0), WEIGHTS)[7] == pytest.approx(0.2 / 0.5)


def test_colour_jaccard():
    features = small_features()
    score, present = colour_scores(features, 0)
    assert score[2] == 1.0
    assert not present[5]                         # the kurta has no colour


def test_missing_attribute_does_not_lower_the_score():
    scores = {
        "text": (np.array([1.0, 1.0]), np.array([True, True])),
        "brand": (np.array([0.0, 0.0]), np.array([False, True])),
    }
    assert combine(scores, {"text": 0.5, "brand": 0.5}).tolist() == [1.0, 0.5]


def test_unknown_weight_is_rejected():
    with pytest.raises(ValueError):
        combine(attribute_scores(small_features(), 0), {"colour_of_box": 1.0})


def test_rank_removes_the_query_and_duplicates():
    features = small_features()
    order, _ = rank(features, 0)
    ids = features.ids[order].tolist()
    assert "q" not in ids and "q2" not in ids     # the query and its twin
    assert "s1b" not in ids                       # s1 and s1b are one product; s1 is kept
    assert len(ids) == len(set(features.asin)) - 1


def test_tie_is_broken_by_rating():
    features = small_features()
    order, score = rank(features, 0)
    assert features.ids[order[:2]].tolist() == ["s2", "s1"]
    assert score[0] == score[1]


def test_score_breakdown_adds_up():
    features = small_features()
    table = score_breakdown(features, 0, 5)
    expected = combine(attribute_scores(features, 0), WEIGHTS)[5]
    assert table["contribution"].sum() == pytest.approx(expected)
    assert not table.loc["brand", "present"]


def test_num_similar_must_be_positive():
    with pytest.raises(ValueError):
        find_similar_products("anything", 0)


needs_data = pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="run `python -m similarity.preprocessing` first")


@needs_data
def test_find_similar_products():
    query, twin = "26d41bdc1495de290bc8e6062d927729", "4188c32471f46876ed222dfcae360791"  # same ASIN
    ids = find_similar_products(query, 10)
    assert len(ids) == 10 and all(isinstance(i, str) for i in ids)
    assert query not in ids and twin not in ids
    assert find_similar_products(query, 10) == ids    # same answer every time


@needs_data
def test_unknown_product_id():
    with pytest.raises(KeyError):
        find_similar_products("not-a-product", 5)


@needs_data
def test_asking_for_more_than_exist():
    ids = find_similar_products("26d41bdc1495de290bc8e6062d927729", 100_000)
    assert len(ids) == 29_529 - 1                 # one per ASIN, minus the query's
    assert len(set(ids)) == len(ids)
