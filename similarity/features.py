"""Stage 02: turn the clean products table into numbers that can be compared.

Run from the repo root:

    python -m similarity.features

Input:  data/processed/products.parquet (from stage 01)
Output: a Features object in memory. It builds in about a second, so it is
        not saved; stage 03 builds it at startup.

Row i of every array in Features is the same product as row i of products.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from similarity.embeddings import Embeddings
from similarity.preprocessing import load_products

# Different spellings of the same colour, checked against the data (see process_readme.md).
COLOUR_SPELLINGS = {
    "gray": "grey",
    "navy": "navy blue",
    "multi": "multicolour",
    "multicolor": "multicolour",
    "off-white": "off white",
    "golden": "gold",
}

# "t-shirt", "t shirt", "tshirt", "t-shirts". Without this, the one-letter "t" is
# dropped and a T-shirt reads as a plain "shirt".
TSHIRT_PATTERN = re.compile(r"\bt[\s-]?shirts?\b")

# A child category starts with who it is for: MensT_Shirts, GirlsDressesJumpsuits.
CATEGORY_GENDERS = {
    "Mens": "men",
    "Womens": "women",
    "Maternity": "women",
    "Boys": "boys",
    "Girls": "girls",
}

# Title words for each gender. "women" does not match \bmen\b: there is no word break inside it.
TITLE_GENDERS = {
    "men": re.compile(r"\b(men|mens|man|gents|male)\b"),
    "women": re.compile(r"\b(women|womens|woman|ladies|lady|female)\b"),
    "boys": re.compile(r"\bboys?\b"),
    "girls": re.compile(r"\bgirls?\b"),
}


@dataclass
class Features:
    ids: np.ndarray               # uniq_id of each row
    row_of: dict[str, int]        # uniq_id -> row number
    text: sparse.csr_matrix       # TF-IDF of the name; each non-empty row has length 1
    words: np.ndarray             # the word behind each text column
    brand: np.ndarray             # integer code per brand, -1 = missing
    category: np.ndarray          # integer code per child category, -1 = missing
    browsenode: np.ndarray        # integer code per browsenode, -1 = missing
    gender: np.ndarray            # integer code per gender (men, women, boys, girls), -1 = missing
    colours: sparse.csr_matrix    # 1 where the product comes in that colour
    colour_names: np.ndarray      # the colour behind each colours column
    price: np.ndarray             # rupees, NaN = missing
    weight: np.ndarray            # grams, NaN = missing
    rating: np.ndarray            # rating_adjusted, used to break ties
    asin: np.ndarray              # integer code per ASIN, to find duplicates
    # Stages 06 and 07: read from data/processed/ the first time a method uses them (see search.py)
    title_embeddings: Embeddings | None = None
    image_embeddings: Embeddings | None = None


def prepare_text(name: str) -> str:
    """Lowercase, and write every form of 't-shirt' as 'tshirt'."""
    return TSHIRT_PATTERN.sub("tshirt", name.lower())


def build_text_matrix(names: pd.Series) -> tuple[np.ndarray, sparse.csr_matrix]:
    """TF-IDF over single words. Words found in only one title are dropped: they can never be shared."""
    vectorizer = TfidfVectorizer(preprocessor=prepare_text, min_df=2)
    matrix = vectorizer.fit_transform(names)
    return vectorizer.get_feature_names_out(), matrix.tocsr()


def encode(values: pd.Series) -> np.ndarray:
    """One integer per distinct value, a label only: compare with ==, never with arithmetic.

    Missing values get -1. Note that -1 == -1 is True, so a comparison must skip
    the pair when either side is -1, or two unknown brands would count as a match.
    """
    codes, _ = pd.factorize(values)
    return codes


def derive_gender(category: str | None, name: str) -> str | None:
    """Who the product is for: men, women, boys or girls.

    The category says it first (MensT_Shirts -> men). Otherwise the title decides,
    but only when it names exactly one gender: "Socks for Men and Women" -> None.
    """
    if isinstance(category, str):
        for prefix, gender in CATEGORY_GENDERS.items():
            if category.startswith(prefix):
                return gender
    title = name.lower()
    found = [gender for gender, pattern in TITLE_GENDERS.items() if pattern.search(title)]
    return found[0] if len(found) == 1 else None


def genders(products: pd.DataFrame) -> pd.Series:
    """derive_gender for every product."""
    return pd.Series(
        [derive_gender(category, name) for category, name in zip(products["child_category"], products["name"])],
        index=products.index,
    )


def normalise_colours(colours: list[str] | None) -> list[str] | None:
    """['gray', 'navy'] -> ['grey', 'navy blue']."""
    if colours is None:
        return None
    return sorted({COLOUR_SPELLINGS.get(colour, colour) for colour in colours})


def build_colour_matrix(colours: pd.Series) -> tuple[np.ndarray, sparse.csr_matrix]:
    """One column per colour, 1 where the product comes in that colour."""
    binarizer = MultiLabelBinarizer(sparse_output=True)
    matrix = binarizer.fit_transform(
        normalise_colours(c) or [] for c in colours
    )
    return binarizer.classes_, matrix.tocsr()


def build_features(products: pd.DataFrame) -> Features:
    products = products.reset_index(drop=True)
    words, text = build_text_matrix(products["name"])
    colour_names, colours = build_colour_matrix(products["colours"])
    ids = products["uniq_id"].to_numpy()
    return Features(
        ids=ids,
        row_of={uniq_id: row for row, uniq_id in enumerate(ids)},
        text=text,
        words=words,
        brand=encode(products["brand"]),
        category=encode(products["child_category"]),
        browsenode=encode(products["browsenode"]),
        gender=encode(genders(products)),
        colours=colours,
        colour_names=colour_names,
        price=products["price"].to_numpy(dtype=float),
        weight=products["weight_g"].to_numpy(dtype=float),
        rating=products["rating_adjusted"].to_numpy(dtype=float),
        asin=encode(products["asin"]),
    )


def top_words(features: Features, row: int, n: int = 10) -> pd.Series:
    """The highest-weighted words of one product's title."""
    weights = features.text[row]
    order = np.argsort(weights.data)[::-1][:n]
    return pd.Series(weights.data[order], index=features.words[weights.indices[order]])


def main() -> None:
    start = time.perf_counter()
    products = load_products()
    features = build_features(products)
    seconds = time.perf_counter() - start
    print(f"Built features for {len(features.ids):,} products in {seconds:.1f} s")
    print(f"text:    {features.text.shape[1]:,} words")
    print(f"colours: {features.colours.shape[1]:,} colours")
    print(f"brands:  {features.brand.max() + 1:,}, categories: {features.category.max() + 1}, "
          f"browsenodes: {features.browsenode.max() + 1}")
    print(f"gender:  {(features.gender != -1).sum():,} products known")


if __name__ == "__main__":
    main()
