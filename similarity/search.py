"""Stage 03: find the products most similar to a given one.

Run from the repo root:

    python -m similarity.search <uniq_id> [num_similar]

Every product is scored against the query on each attribute (0 to 1). The
scores are combined into one weighted average over the attributes both products
have, and the best are returned. This is exact search: every product is scored, in about
20 to 40 ms over 30,000 products. The setting SEARCH_MODE=ann scores only an HNSW shortlist
instead (stage 08, ann.py); see `search` below.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from similarity import config
from similarity.embeddings import IMAGE_PATH, TITLE_PATH, Embeddings, load_embeddings, rescale
from similarity.features import Features, build_features
from similarity.preprocessing import load_products

# How much each attribute counts (settings WEIGHT_*: see similarity/config.py and .env.example).
# A score only divides by the weights of the attributes a pair actually has, so these do not
# need to add up to 1. These are the weights of method "tfidf".
WEIGHTS = {
    "text": config.WEIGHT_TEXT,
    "category": config.WEIGHT_CATEGORY,
    "gender": config.WEIGHT_GENDER,
    "price": config.WEIGHT_PRICE,
    "brand": config.WEIGHT_BRAND,
    "colour": config.WEIGHT_COLOUR,
    "weight": config.WEIGHT_WEIGHT,
}


# The photo score of stage 07, added on top of the other weights.
IMAGE_WEIGHT = config.WEIGHT_IMAGE


def method_weights(title: str = "tfidf", image: bool = False) -> dict[str, float]:
    """The weights of one method. "minilm" swaps the TF-IDF title score for the sentence-transformer
    one (stage 06) at the same weight; image=True adds the photo score (stage 07). Everything else
    stays as it is."""
    if title == "tfidf":
        weights = dict(WEIGHTS)
    elif title == "minilm":
        weights = {("text_embedding" if name == "text" else name): weight for name, weight in WEIGHTS.items()}
    else:
        raise ValueError(f"unknown title score: {title}")
    if image:
        weights["image"] = IMAGE_WEIGHT
    return weights


# The methods: the API's setting SIMILARITY_METHOD picks one; evals/run_eval.py compares them all.
METHOD_WEIGHTS = {
    "tfidf": method_weights("tfidf"),
    "minilm": method_weights("minilm"),
    "tfidf_image": method_weights("tfidf", image=True),
    "minilm_image": method_weights("minilm", image=True),
}

# Scores are rounded to this many decimals before sorting, so 0.8123 and 0.8119
# count as a tie and the rating decides.
SCORE_DECIMALS = config.SCORE_DECIMALS


# --- one score per attribute -------------------------------------------------------
# Each function compares the query row with every product at once and returns two
# arrays: the score (0 to 1) and whether the pair could be compared at all.

def text_scores(features: Features, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of the TF-IDF rows.

    Never missing: every product has a title. A title with no vector shares no
    word with any other title, so 0 is its true score, not an unknown one.
    """
    score = (features.text @ features.text[row].T).toarray().ravel()
    return score, np.ones(len(score), dtype=bool)


def title_embeddings(features: Features) -> Embeddings:
    """The stage 06 title numbers: read on first use, then kept with the features.
    Only a method that weighs `text_embedding` ever reads them, so the API never does."""
    if features.title_embeddings is None:
        features.title_embeddings = load_embeddings(TITLE_PATH, features.ids)
    return features.title_embeddings


def embedding_scores(embeddings: Embeddings, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of the embedding rows, rescaled so a typical random pair scores 0 (see embeddings.rescale).
    Missing when either product has nothing embedded (no photo, in stage 07)."""
    cosine = embeddings.vectors @ embeddings.vectors[row]
    return rescale(cosine, embeddings.baseline), embeddings.present & embeddings.present[row]


def text_embedding_scores(features: Features, row: int) -> tuple[np.ndarray, np.ndarray]:
    """The sentence-transformer title score of stage 06. Never missing: every product has a title."""
    return embedding_scores(title_embeddings(features), row)


def image_embeddings(features: Features) -> Embeddings:
    """The stage 07 photo numbers: read on first use, then kept with the features."""
    if features.image_embeddings is None:
        features.image_embeddings = load_embeddings(IMAGE_PATH, features.ids)
    return features.image_embeddings


def image_scores(features: Features, row: int) -> tuple[np.ndarray, np.ndarray]:
    """The CLIP photo score of stage 07. Missing when either product has no photo, so the pair is
    judged on its other attributes alone (the fallback)."""
    return embedding_scores(image_embeddings(features), row)


def code_scores(codes: np.ndarray, row: int) -> tuple[np.ndarray, np.ndarray]:
    """1 when the codes are equal, else 0. Missing when either side is -1."""
    score = (codes == codes[row]).astype(float)
    present = (codes != -1) & (codes[row] != -1)
    return score, present


def category_scores(features: Features, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Compare child categories; where a pair cannot, compare browsenodes instead."""
    category, has_category = code_scores(features.category, row)
    node, has_node = code_scores(features.browsenode, row)
    return np.where(has_category, category, node), has_category | has_node


def colour_scores(features: Features, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Jaccard: colours in common ÷ colours of either. Missing when either has no colour."""
    count = features.colours.getnnz(axis=1)
    shared = (features.colours @ features.colours[row].T).toarray().ravel()
    either = count + count[row] - shared
    score = np.divide(shared, either, out=np.zeros(len(count)), where=either > 0)
    return score, (count > 0) & (count[row] > 0)


def ratio_scores(values: np.ndarray, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Smaller ÷ larger: ₹500 vs ₹1,000 scores 0.5, like ₹5,000 vs ₹10,000. Missing when either is NaN."""
    score = np.minimum(values, values[row]) / np.maximum(values, values[row])
    present = ~np.isnan(values) & ~np.isnan(values[row])
    return np.nan_to_num(score), present


# Each attribute's scoring function, by the name used in the weights.
SCORERS = {
    "text": text_scores,
    "text_embedding": text_embedding_scores,
    "image": image_scores,
    "category": category_scores,
    "gender": lambda features, row: code_scores(features.gender, row),
    "price": lambda features, row: ratio_scores(features.price, row),
    "brand": lambda features, row: code_scores(features.brand, row),
    "colour": colour_scores,
    "weight": lambda features, row: ratio_scores(features.weight, row),
}


def attribute_scores(features: Features, row: int, names=WEIGHTS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(score, present) of the query row against all products, for each attribute in `names`.
    Only the attributes asked for are computed, so a method never reads files it does not use."""
    return {name: SCORERS[name](features, row) for name in names if name in SCORERS}


# --- combine and rank ----------------------------------------------------------------

def combine(scores: dict[str, tuple[np.ndarray, np.ndarray]], weights: dict[str, float]) -> np.ndarray:
    """Weighted average over the attributes each pair has: sum(w * s * present) / sum(w * present).

    An attribute missing on either side is left out instead of counting as 0
    (Gower's similarity coefficient, 1971). A pair with nothing to compare scores 0.
    """
    unknown = set(weights) - set(scores)
    if unknown:
        raise ValueError(f"unknown attributes in weights: {sorted(unknown)}")
    size = len(next(iter(scores.values()))[0])
    total = np.zeros(size)
    used = np.zeros(size)
    for name, weight in weights.items():
        score, present = scores[name]
        total += weight * score * present
        used += weight * present
    return np.divide(total, used, out=np.zeros(size), where=used > 0)


def rank(features: Features, row: int, weights: dict[str, float] = WEIGHTS) -> tuple[np.ndarray, np.ndarray]:
    """The rows of all other products, most similar first, and their scores.

    Removed: the query and every row with its ASIN (the same product listed twice),
    and all but the best row of any other ASIN.
    Order: score (rounded), then adjusted rating, then uniq_id, so the answer never changes.
    """
    score = combine(attribute_scores(features, row, weights), weights)
    rounded = np.round(score, SCORE_DECIMALS)
    order = np.lexsort((features.ids, -features.rating, -rounded))   # last key sorts first
    order = order[features.asin[order] != features.asin[row]]
    _, first = np.unique(features.asin[order], return_index=True)   # best row of each ASIN
    order = order[np.sort(first)]
    return order, score[order]


def score_breakdown(features: Features, a: int, b: int, weights: dict[str, float] = WEIGHTS) -> pd.DataFrame:
    """Why b scores what it does against a: each attribute's score and its part of the total."""
    scores = attribute_scores(features, a, weights)
    table = pd.DataFrame(
        {
            "weight": weights,
            "present": {name: bool(scores[name][1][b]) for name in weights},
            "score": {name: scores[name][0][b] if scores[name][1][b] else np.nan for name in weights},
        }
    )
    used = table["weight"] * table["present"]
    table["rescaled_weight"] = used / used.sum()
    table["contribution"] = (table["rescaled_weight"] * table["score"]).fillna(0)
    return table


# --- checks without labels ----------------------------------------------------------

def result_checks(features: Features, rows: list[int], weights: dict[str, float] = WEIGHTS, k: int = 10) -> pd.DataFrame:
    """For each query row, how its top k compare with it on attributes that are easy to judge.

    same_category, same_gender: share of the top k that match, where both are known
    price_ratio: median smaller ÷ larger price over the top k
    no_shared_words: share of the top k whose title shares no word with the query's
    evidence_top, evidence_all: average total weight a pair was judged on, for the
        top k and for all products. If the top k were judged on much less than
        average, missing attributes are pushing products up.
    """
    records = []
    for row in rows:
        order, _ = rank(features, row, weights)
        top = order[:k]
        scores = attribute_scores(features, row)
        evidence = sum(WEIGHTS[name] * scores[name][1] for name in WEIGHTS)

        def match(name):
            score, present = scores[name]
            known = top[present[top]]
            return score[known].mean() if len(known) else np.nan

        price, has_price = scores["price"]
        priced = top[has_price[top]]
        records.append(
            {
                "row": row,
                "same_category": match("category"),
                "same_gender": match("gender"),
                "price_ratio": np.median(price[priced]) if len(priced) else np.nan,
                "no_shared_words": (scores["text"][0][top] == 0).mean(),
                "evidence_top": evidence[top].mean(),
                "evidence_all": np.delete(evidence, row).mean(),
            }
        )
    return pd.DataFrame(records).set_index("row")


# --- the function the problem statement asks for -------------------------------------------------

@lru_cache(maxsize=1)
def load_features() -> Features:
    """Build the features on the first call; later calls reuse them."""
    return build_features(load_products())


def search(features: Features, row: int, method: str | None = None, mode: str | None = None):
    """(rows, scores) of the products most similar to `row`, best first.

    method  which score: tfidf, minilm, tfidf_image or minilm_image (default: setting SIMILARITY_METHOD)
    mode    exact: score every product; ann: an HNSW shortlist, then the same score on it
            (stage 8; needs faiss-cpu and the index) (default: setting SEARCH_MODE)
    """
    method = method or config.SIMILARITY_METHOD
    mode = mode or config.SEARCH_MODE
    if method not in METHOD_WEIGHTS:
        raise ValueError(f"unknown method {method!r}: choose one of {', '.join(METHOD_WEIGHTS)}")
    if mode == "exact":
        return rank(features, row, METHOD_WEIGHTS[method])
    if mode == "ann":
        from similarity.ann import ann_rank   # faiss is imported only when ANN is asked for

        return ann_rank(features, row, METHOD_WEIGHTS[method])
    raise ValueError(f"unknown search mode {mode!r}: choose one of {', '.join(config.SEARCH_MODES)}")


def find_similar_products(product_id: str, num_similar: int = 10,
                          method: str | None = None, mode: str | None = None) -> list[str]:
    """The uniq_ids of the num_similar products most similar to product_id, most similar first.
    `method` and `mode` default to the settings SIMILARITY_METHOD and SEARCH_MODE (see `search`).

    Raises KeyError for an unknown product_id and ValueError when num_similar < 1.
    Asking for more products than exist returns all of them.
    """
    if num_similar < 1:
        raise ValueError(f"num_similar must be at least 1, got {num_similar}")
    features = load_features()
    if product_id not in features.row_of:
        raise KeyError(f"unknown product_id: {product_id}")
    order, _ = search(features, features.row_of[product_id], method, mode)
    return features.ids[order[:num_similar]].tolist()


def main() -> None:
    products = load_products()
    product_id = sys.argv[1] if len(sys.argv) > 1 else products["uniq_id"].iloc[0]
    num_similar = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    ids = find_similar_products(product_id, num_similar)
    columns = ["uniq_id", "name", "child_category", "price"]
    by_id = products.set_index("uniq_id", drop=False)
    print("Query:")
    print(by_id.loc[[product_id], columns].to_string(index=False))
    print(f"\nTop {len(ids)}:")
    print(by_id.loc[ids, columns].to_string(index=False))


if __name__ == "__main__":
    main()
