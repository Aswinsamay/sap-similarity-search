"""Stage 01: clean the raw Amazon fashion file into one typed table.

Run from the repo root:

    python -m similarity.preprocessing

Input:  data/archive/marketing_sample_...ldjson  (30,000 raw products)
Output: data/processed/products.parquet           (30,000 clean products)

This stage only makes each field clean and typed. It does not decide how
products are compared; that is the job of the later stages. Every rule here
comes from a finding in process_readme.md. Missing values are stored as null,
never guessed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path(
    "data/archive/"
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)
PROCESSED_PATH = Path("data/processed/products.parquet")

# The only raw columns this project uses. Everything else is out of scope (see process_readme.md).
RAW_COLUMNS = [
    "uniq_id",
    "asin",
    "product_name",
    "brand",
    "colour",
    "sales_price",
    "weight",
    "rating",
    "no__of_reviews",
    "product_details__k_v_pairs",
    "sales_rank_in_child_category",
    "browsenode",
    "medium",
]

PLACEHOLDER_BRANDS = {"generic", "n/a"}
WEIGHT_SENTINEL = "999999999"
MAX_WEIGHT_G = 20_000      # no garment weighs 20 kg; above this the unit is wrong
PLACEHOLDER_WEIGHT_G = 4.54  # 0.01 lb converted to grams, entered as a placeholder
REVIEW_PRIOR_WEIGHT = 5    # the prior in the Bayesian rating counts as 5 reviews

# "200 g", "1.5 Kg", "150 Grams", "2 Kilograms", "120 �grams"
WEIGHT_PATTERN = re.compile(r"^([\d.]+)\W*(kg|kilograms?|g|grams?)$", re.IGNORECASE)
# "3.7 out of 5 stars 23 customer reviews", "5.0 out of 5 stars 1 customer review"
REVIEWS_PATTERN = re.compile(r"([\d,]+) customer review")
NO_REVIEWS_TEXT = "Be the first to review this item"


# --- field parsers: one raw value in, one clean value (or None) out ---------------

def clean_name(value: object) -> str:
    """Collapse runs of whitespace. Lowercasing is left to the text stage."""
    return " ".join(str(value).split())


def parse_brand(value: object) -> str | None:
    """'  BIBA ' -> 'biba'. Placeholders such as 'Generic' -> None."""
    if not isinstance(value, str):
        return None
    brand = " ".join(value.split()).lower()
    if brand == "" or brand in PLACEHOLDER_BRANDS:
        return None
    return brand


def parse_colours(value: object) -> list[str] | None:
    """'Black|Navy Blue|black' -> ['black', 'navy blue']. Sorted, no duplicates."""
    if not isinstance(value, str):
        return None
    colours = {part.strip().lower() for part in value.split("|")}
    colours.discard("")
    return sorted(colours) or None


def parse_price(value: object) -> float | None:
    """Keep positive prices only; a price of 0 or less cannot be real."""
    price = pd.to_numeric(value, errors="coerce")
    if pd.isna(price) or price <= 0:
        return None
    return float(price)


def parse_weight_grams(value: object) -> float | None:
    """'1.5 Kg' -> 1500.0. The sentinel, unparseable text and implausible values -> None."""
    if not isinstance(value, str) or value.strip() == WEIGHT_SENTINEL:
        return None
    match = WEIGHT_PATTERN.match(value.strip())
    if match is None:
        return None
    number_text, unit = match.groups()
    try:
        number = float(number_text)
    except ValueError:  # e.g. "1.2.3"
        return None
    grams = number * 1000 if unit.lower().startswith("k") else number
    if grams <= 0 or grams > MAX_WEIGHT_G or round(grams, 2) == PLACEHOLDER_WEIGHT_G:
        return None
    return grams


def parse_rating(value: object) -> float | None:
    rating = pd.to_numeric(value, errors="coerce")
    if pd.isna(rating) or not 1 <= rating <= 5:
        return None
    return float(rating)


def parse_review_count(details: object, reviews_column: object) -> float | None:
    """Number of reviews, from the best source available.

    1. The count in the details text ("23 customer reviews").
    2. The no__of_reviews column.
    3. "Be the first to review this item" means 0.
    Otherwise unknown (None).
    """
    text = details.get("Customer_Reviews", "") if isinstance(details, dict) else ""
    match = REVIEWS_PATTERN.search(text)
    if match:
        return float(match.group(1).replace(",", ""))
    column_count = pd.to_numeric(reviews_column, errors="coerce")
    if not pd.isna(column_count):
        return float(column_count)
    if text.startswith(NO_REVIEWS_TEXT):
        return 0.0
    return None


def parse_category(value: object) -> str | None:
    """{'MensT_Shirts': '#12151'} -> 'MensT_Shirts'. The rank is popularity, so it is dropped."""
    if isinstance(value, dict) and value:
        return str(next(iter(value)))
    return None


def parse_browsenode(value: object) -> str | None:
    """1968255031.0 -> '1968255031'. An ID, so it is stored as text, not as a number."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return str(int(number))


def parse_first_url(value: object) -> str | None:
    """'a.jpg|b.jpg' -> 'a.jpg'. The first image is the main product photo."""
    if not isinstance(value, str):
        return None
    return value.split("|")[0].strip() or None


# --- derived field ----------------------------------------------------------------

def adjusted_rating(
    rating: pd.Series, review_count: pd.Series, prior_weight: float = REVIEW_PRIOR_WEIGHT
) -> pd.Series:
    """Bayesian average rating, used only to break ties.

        adjusted = (prior_weight * prior_mean + n * rating) / (prior_weight + n)

    prior_mean is the average star over every review in the dataset. A rating
    from few reviews is pulled toward it; a rating from many reviews keeps its
    value. An unknown review count counts as 0 reviews, so the product gets
    prior_mean.
    """
    n = review_count.fillna(0)
    prior_mean = (rating * n).sum() / n.sum()
    return (prior_weight * prior_mean + n * rating) / (prior_weight + n)


# --- the stage --------------------------------------------------------------------

def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_json(path, lines=True)[RAW_COLUMNS]


def build_products(raw: pd.DataFrame) -> pd.DataFrame:
    """Run every parser over the raw table. One output row per input row."""
    products = pd.DataFrame(
        {
            "uniq_id": raw["uniq_id"].astype(str),
            "asin": raw["asin"].astype(str),
            "name": raw["product_name"].map(clean_name),
            "brand": raw["brand"].map(parse_brand),
            "colours": raw["colour"].map(parse_colours),
            "price": raw["sales_price"].map(parse_price).astype(float),
            "weight_g": raw["weight"].map(parse_weight_grams).astype(float),
            "rating": raw["rating"].map(parse_rating).astype(float),
            "review_count": [
                parse_review_count(details, column)
                for details, column in zip(raw["product_details__k_v_pairs"], raw["no__of_reviews"])
            ],
            "child_category": raw["sales_rank_in_child_category"].map(parse_category),
            "browsenode": raw["browsenode"].map(parse_browsenode),
            "image_url": raw["medium"].map(parse_first_url),
        }
    )
    products["review_count"] = products["review_count"].astype(float)
    products["rating_adjusted"] = adjusted_rating(products["rating"], products["review_count"])
    return products


def validate(products: pd.DataFrame) -> None:
    """Raise ValueError listing every check that fails."""
    price = products["price"].dropna()
    weight = products["weight_g"].dropna()
    checks = {
        "uniq_id is unique": products["uniq_id"].is_unique,
        "uniq_id is never empty": (products["uniq_id"].str.len() > 0).all(),
        "every product has a name": (products["name"].str.len() > 0).all(),
        "prices are positive": (price > 0).all(),
        f"weights are in (0, {MAX_WEIGHT_G}] g": ((weight > 0) & (weight <= MAX_WEIGHT_G)).all(),
        "ratings are in [1, 5]": products["rating"].dropna().between(1, 5).all(),
        "adjusted ratings are in [1, 5]": products["rating_adjusted"].dropna().between(1, 5).all(),
        "review counts are not negative": (products["review_count"].dropna() >= 0).all(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"products failed validation: {failed}")


def save(products: pd.DataFrame, path: str | Path = PROCESSED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    products.to_parquet(path, index=False)
    return path


def load_products(path: str | Path = PROCESSED_PATH) -> pd.DataFrame:
    """Read the processed table back. Parquet returns lists as arrays; turn them back into lists."""
    products = pd.read_parquet(path)
    products["colours"] = products["colours"].map(
        lambda colours: list(colours) if colours is not None else None
    )
    return products


def fill_summary(products: pd.DataFrame) -> pd.DataFrame:
    """Non-null count and percentage for every column."""
    filled = products.notna().sum()
    return pd.DataFrame({"filled": filled, "percent": (100 * filled / len(products)).round(1)})


def main() -> None:
    raw = load_raw()
    print(f"Loaded {len(raw):,} raw rows from {RAW_PATH}")
    products = build_products(raw)
    validate(products)
    path = save(products)
    print(f"Wrote {path} ({len(products):,} rows, {products.shape[1]} columns)")
    print(fill_summary(products).to_string())


if __name__ == "__main__":
    main()
