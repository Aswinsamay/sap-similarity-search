import pandas as pd
import pytest

from similarity.preprocessing import (
    RAW_PATH,
    adjusted_rating,
    build_products,
    clean_name,
    load_products,
    load_raw,
    parse_brand,
    parse_browsenode,
    parse_category,
    parse_colours,
    parse_first_url,
    parse_price,
    parse_rating,
    parse_review_count,
    parse_weight_grams,
    save,
    validate,
)


def test_clean_name():
    assert clean_name("  Men's   Cotton\tT-Shirt ") == "Men's Cotton T-Shirt"


@pytest.mark.parametrize(
    "raw, expected",
    [("  BIBA ", "biba"), ("LA' Facon", "la' facon"), ("Generic", None), ("N/A", None), (None, None)],
)
def test_parse_brand(raw, expected):
    assert parse_brand(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Black|Navy Blue", ["black", "navy blue"]),
        ("White|Black|black", ["black", "white"]),
        ("|", None),
        (None, None),
    ],
)
def test_parse_colours(raw, expected):
    assert parse_colours(raw) == expected


@pytest.mark.parametrize("raw, expected", [(200.0, 200.0), (0.0, None), (-5, None), (None, None)])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    "raw, grams",
    [
        ("200 g", 200.0),
        ("99.8 g", 99.8),
        ("150 Grams", 150.0),
        ("1.5 Kg", 1500.0),
        ("2 Kilograms", 2000.0),
        ("120 �grams", 120.0),
    ],
)
def test_parse_weight_units(raw, grams):
    assert parse_weight_grams(raw) == pytest.approx(grams)


@pytest.mark.parametrize(
    "raw",
    [
        "999999999",              # sentinel
        "The Slim 3 Lift shirt",  # a sentence, not a weight
        "113 Kilograms",          # unit error: a stocking does not weigh 113 kg
        "4.54 g",                 # 0.01 lb placeholder
        "1.2.3 g",
        None,
    ],
)
def test_parse_weight_missing(raw):
    assert parse_weight_grams(raw) is None


@pytest.mark.parametrize("raw, expected", [(4.5, 4.5), (0.0, None), (6.0, None), (None, None)])
def test_parse_rating(raw, expected):
    assert parse_rating(raw) == expected


@pytest.mark.parametrize(
    "details, column, expected",
    [
        ({"Customer_Reviews": "3.7 out of 5 stars 23 customer reviews"}, None, 23.0),
        ({"Customer_Reviews": "5.0 out of 5 stars 1 customer review"}, None, 1.0),
        ({"Customer_Reviews": "4.1 out of 5 stars 1,204 customer reviews"}, None, 1204.0),
        ({"Customer_Reviews": "3.7 out of 5 stars 23 customer reviews"}, 40.0, 23.0),  # text wins
        ({"Customer_Reviews": "Be the first to review this item"}, 39.0, 39.0),       # then column
        ({"Customer_Reviews": "Be the first to review this item"}, None, 0.0),        # then zero
        (None, None, None),
    ],
)
def test_parse_review_count(details, column, expected):
    assert parse_review_count(details, column) == expected


def test_parse_category_browsenode_url():
    assert parse_category({"MensT_Shirts": "#12151"}) == "MensT_Shirts"
    assert parse_category(None) is None
    assert parse_browsenode(1968255031.0) == "1968255031"
    assert parse_browsenode(float("nan")) is None
    assert parse_first_url("a.jpg|b.jpg") == "a.jpg"
    assert parse_first_url(None) is None


def test_adjusted_rating_pulls_small_samples_to_the_mean():
    rating = pd.Series([5.0, 4.4, 3.5, 3.0])
    reviews = pd.Series([1.0, 200.0, 300.0, None])
    adjusted = adjusted_rating(rating, reviews, prior_weight=5)
    prior_mean = (5.0 * 1 + 4.4 * 200 + 3.5 * 300) / 501   # every review counts once
    assert adjusted[0] == pytest.approx((5 * prior_mean + 5.0) / 6)
    assert adjusted[1] > adjusted[0]                  # many reviews beat one perfect review
    assert adjusted[3] == pytest.approx(prior_mean)   # unknown count -> the prior


def test_validate_rejects_duplicate_ids():
    products = pd.DataFrame(
        {
            "uniq_id": ["a", "a"],
            "name": ["x", "y"],
            "price": [1.0, 2.0],
            "weight_g": [None, None],
            "rating": [4.0, 4.0],
            "rating_adjusted": [4.0, 4.0],
            "review_count": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="uniq_id is unique"):
        validate(products)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="dataset not present")
def test_full_run_matches_eda(tmp_path):
    products = build_products(load_raw())
    validate(products)

    assert len(products) == 30_000                         # every row kept, duplicates too
    assert products["brand"].notna().sum() == 21_857 - 245 - 4
    assert products["colours"].notna().sum() == 6_029
    assert products["price"].notna().sum() == 27_110
    assert products["weight_g"].notna().sum() == 6_254 - 8 - 62
    assert products["child_category"].notna().sum() == 24_851
    assert products["browsenode"].notna().sum() == 29_480
    assert products["image_url"].notna().sum() == 29_998

    path = save(products, tmp_path / "products.parquet")
    reloaded = load_products(path)
    pd.testing.assert_frame_equal(reloaded, products)
