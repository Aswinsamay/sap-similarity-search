import numpy as np
import pandas as pd
import pytest

from similarity.features import (
    build_colour_matrix,
    build_features,
    build_text_matrix,
    derive_gender,
    encode,
    normalise_colours,
    prepare_text,
    top_words,
)
from similarity.preprocessing import PROCESSED_PATH, load_products


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Men's Printed T-Shirt", "men's printed tshirt"),
        ("Boys T Shirts Pack", "boys tshirt pack"),
        ("Cotton Tshirt", "cotton tshirt"),
        ("Casual Shirt", "casual shirt"),
        ("Kit-Shirt Set", "kit-shirt set"),  # 't' inside a word is left alone
    ],
)
def test_prepare_text(name, expected):
    assert prepare_text(name) == expected


def test_normalise_colours():
    assert normalise_colours(["gray", "black"]) == ["black", "grey"]
    assert normalise_colours(["navy", "navy blue"]) == ["navy blue"]
    assert normalise_colours(None) is None


@pytest.mark.parametrize(
    "category, name, expected",
    [
        ("MensT_Shirts", "Printed Tee", "men"),
        ("MaternityTops", "Cotton Top", "women"),
        ("GirlsT_Shirts", "Men's Tee", "girls"),       # the category wins over the title
        ("Fashion", "Boys Shorts", "boys"),            # category without a gender: the title decides
        (None, "Women's Kurta", "women"),
        (None, "Men's T-Shirt", "men"),
        (None, "Socks for Men and Women", None),       # two genders named
        (None, "Cotton Saree", None),                  # none named
    ],
)
def test_derive_gender(category, name, expected):
    assert derive_gender(category, name) == expected


def test_encode_gives_missing_minus_one():
    assert encode(pd.Series(["a", None, "b", "a"])).tolist() == [0, -1, 1, 0]


def test_text_matrix():
    names = pd.Series(["Men's T-Shirt", "Women's Tshirt", "Cotton Kurta", "Silk Kurta"])
    words, matrix = build_text_matrix(names)
    assert "tshirt" in words
    assert "silk" not in words                         # in one title only
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)                     # every row has length 1
    similarity = (matrix @ matrix.T).toarray()
    assert similarity[0, 1] > 0                        # T-Shirt and Tshirt now share a word
    assert similarity[0, 2] == 0


def test_colour_matrix():
    names, matrix = build_colour_matrix(pd.Series([["black", "gray"], None, ["grey"]]))
    assert names.tolist() == ["black", "grey"]
    assert matrix.toarray().tolist() == [[1, 1], [0, 0], [0, 1]]


def small_products():
    return pd.DataFrame(
        {
            "uniq_id": ["a", "b", "c"],
            "asin": ["X", "X", "Y"],
            "name": ["Men's T-Shirt", "Men's Shirt", "Women's Kurta"],
            "brand": ["max", None, "max"],
            "colours": [["black"], None, ["gray"]],
            "price": [200.0, np.nan, 500.0],
            "weight_g": [np.nan, 150.0, np.nan],
            "child_category": ["MensT_Shirts", "MensCasualShirts", None],
            "browsenode": ["1", "2", "3"],
            "rating_adjusted": [4.0, 4.2, 3.9],
        }
    )


def test_build_features_rows_line_up():
    features = build_features(small_products())
    assert features.ids.tolist() == ["a", "b", "c"]
    assert features.row_of == {"a": 0, "b": 1, "c": 2}
    assert features.text.shape[0] == 3
    assert features.brand.tolist() == [0, -1, 0]
    assert features.category.tolist() == [0, 1, -1]
    assert features.gender.tolist() == [0, 0, 1]       # men, men, women (from the title)
    assert features.asin.tolist() == [0, 0, 1]
    assert np.isnan(features.price[1]) and features.price[2] == 500.0
    assert features.colour_names.tolist() == ["black", "grey"]


def test_top_words():
    features = build_features(small_products())
    assert set(top_words(features, 0).index) <= {"men", "tshirt"}


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="run `python -m similarity.preprocessing` first")
def test_full_features():
    products = load_products()
    features = build_features(products)
    assert features.text.shape == (30_000, 7_412)
    assert (features.brand == -1).sum() == 30_000 - 21_608
    assert (features.category == -1).sum() == 30_000 - 24_851
    assert features.colours.shape[1] == 7_819
    assert "gray" not in set(features.colour_names)
    assert (features.gender == -1).sum() == 2_654
