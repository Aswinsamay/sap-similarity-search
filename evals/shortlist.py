"""Pick the 20 query products and, for each, a shortlist of candidates for the LLM judge to label.

Run once from the repo root:

    python -m evals.shortlist

Writes the Queries and Labels sheets of evals/eval_results.xlsx (see evals/store.py). Labels has
one row per candidate, with `good` and `gt_rank` left empty to be filled in by the judge. It refuses
to run when the workbook already holds labels, so finished labels are never lost.

Why several routes: the reviewer cannot read 30,000 products per query, only a shortlist. If the
shortlist came from the current method alone, the ground truth could only contain what that
method already finds, and the method could never be caught missing a better product. So the
candidates come from four routes, and are shuffled so the reviewer cannot tell which found which:

    current method        top 15
    title words only      top 15   (TF-IDF, stage 02)
    parts of words        top 10   (character TF-IDF: links kurta/kurti, tee/t-shirt, frok/frock)
    same group            10 random products of the same category and gender at a similar price
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from evals.store import WORKBOOK_PATH, has_data, save_data
from similarity.features import genders, prepare_text
from similarity.preprocessing import load_products
from similarity.search import load_features, rank

SEED = 7

# 12 random queries, spread over the largest categories, plus 2 from any other category
RANDOM_PER_CATEGORY = {
    "WomensKurtasKurtis": 2,
    "MensT_Shirts": 2,
    "WomensSarees": 1,
    "MensFormalShirts": 1,
    "GirlsDressesJumpsuits": 1,
    "MensCasualShirts": 1,
    "BoysT_Shirts": 1,
    "WomensDresses": 1,
}
RANDOM_OTHER = 2

# 8 deliberately chosen cases where a search is likely to struggle
HARD_CASES = {
    "26d41bdc1495de290bc8e6062d927729": "saree blouse fabric filed under kurtis",
    "2304753b72be21b48128f7d2ae482c9f": "frock: similar items are often called 'dress'",
    "0f8a3a26f3e563621eda640aa3ecd17e": "kurta: similar items are often called 'kurti'",
    "7fecf1a590d06efc43f720597f479065": "only 2 attributes known (no category, no price)",
    "3ff4c7573b9e673b6ca432608c1969e5": "no category and no price",
    "8e724efa4103309725237750410b6a2a": "misspelled title ('Frok') and no price",
    "45a59081eda442a2d35efb62b936e3b1": "the title is only the brand ('OOMPH!')",
    "981b995e1e8016ef7c17b98b782aa88f": "the same product listed under 3 uniq_ids",
}

ROUTE_SIZES = {"current": 15, "title_words": 15, "word_parts": 10, "same_group": 10}


def select_queries(products: pd.DataFrame) -> pd.DataFrame:
    """The 20 queries: 12 random (fixed seed) and the 8 hard cases."""
    hard = products["uniq_id"].isin(HARD_CASES)
    picked = []
    for category, count in RANDOM_PER_CATEGORY.items():
        pool = products[(products["child_category"] == category) & ~hard]
        picked += [(row, f"random from {category}") for row in pool.sample(count, random_state=SEED).index]
    other = products[products["child_category"].notna() & ~products["child_category"].isin(RANDOM_PER_CATEGORY) & ~hard]
    picked += [(row, f"random from {other.loc[row, 'child_category']}") for row in other.sample(RANDOM_OTHER, random_state=SEED).index]
    row_of = pd.Series(products.index, index=products["uniq_id"])
    picked += [(row_of[uniq_id], why) for uniq_id, why in HARD_CASES.items()]

    queries = pd.DataFrame(
        {
            "query_id": [f"Q{i:02d}" for i in range(1, len(picked) + 1)],
            "uniq_id": [products.loc[row, "uniq_id"] for row, _ in picked],
            "kind": ["random"] * (len(picked) - len(HARD_CASES)) + ["hard"] * len(HARD_CASES),
            "why": [why for _, why in picked],
        }
    )
    assert queries["uniq_id"].is_unique
    return queries


def word_part_matrix(products: pd.DataFrame):
    """Character 3- to 5-grams of each title, so similar spellings share features."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    return vectorizer.fit_transform(products["name"].map(prepare_text))


def same_group(features, row: int, rng: np.random.Generator, size: int) -> list[int]:
    """Random products of the query's category (or browsenode) and gender, at 0.5x to 2x its price."""
    if features.category[row] != -1:
        mask = features.category == features.category[row]
    elif features.browsenode[row] != -1:
        mask = features.browsenode == features.browsenode[row]
    else:
        return []
    if features.gender[row] != -1:
        mask &= features.gender == features.gender[row]
    if not np.isnan(features.price[row]):
        ratio = features.price / features.price[row]
        mask &= (ratio >= 0.5) & (ratio <= 2)
    mask &= features.asin != features.asin[row]
    rows = np.flatnonzero(mask)
    return rng.choice(rows, min(size, len(rows)), replace=False).tolist()


def shortlist(features, row: int, word_parts, rng: np.random.Generator) -> list[int]:
    """About 40 candidate rows for one query, one per ASIN, in random order."""
    current, _ = rank(features, row)
    title_words, _ = rank(features, row, {"text": 1.0})
    part_scores = (word_parts @ word_parts[row].T).toarray().ravel()
    parts, part_asins = [], {features.asin[row]}
    for candidate in np.argsort(-part_scores, kind="stable"):
        if features.asin[candidate] not in part_asins:
            part_asins.add(features.asin[candidate])
            parts.append(int(candidate))
        if len(parts) == ROUTE_SIZES["word_parts"]:
            break

    candidates = (
        current[: ROUTE_SIZES["current"]].tolist()
        + title_words[: ROUTE_SIZES["title_words"]].tolist()
        + parts
        + same_group(features, row, rng, ROUTE_SIZES["same_group"])
    )
    seen, unique = set(), []
    for candidate in candidates:
        if features.asin[candidate] not in seen:
            seen.add(features.asin[candidate])
            unique.append(candidate)
    rng.shuffle(unique)
    return unique


def main() -> None:
    if has_data():
        raise SystemExit(f"{WORKBOOK_PATH} already holds labels; move it away first to rebuild (they would be lost)")
    products = load_products()
    features = load_features()
    gender = genders(products)
    word_parts = word_part_matrix(products)
    rng = np.random.default_rng(SEED)

    queries = select_queries(products)
    rows = []
    for query in queries.itertuples():
        row = features.row_of[query.uniq_id]
        for position, candidate in enumerate(shortlist(features, row, word_parts, rng), start=1):
            rows.append({"query_id": query.query_id, "position": position, "row": candidate})
    labels = pd.DataFrame(rows)
    info = products.loc[labels["row"]]
    labels = labels.assign(
        uniq_id=info["uniq_id"].values,
        name=info["name"].values,
        category=info["child_category"].values,
        gender=gender.loc[labels["row"]].values,
        price=info["price"].values,
        brand=info["brand"].values,
        colours=info["colours"].map(lambda c: "|".join(c) if c else None).values,
        good=None,
        gt_rank=None,
        source="shortlist",
    ).drop(columns="row")

    query_info = products.set_index("uniq_id").loc[queries["uniq_id"]]
    queries = queries.assign(
        name=query_info["name"].values,
        category=query_info["child_category"].values,
        gender=gender.loc[[features.row_of[u] for u in queries["uniq_id"]]].values,
        price=query_info["price"].values,
    )

    WORKBOOK_PATH.parent.mkdir(exist_ok=True)
    save_data(queries, labels)
    print(f"Wrote the Queries ({len(queries)}) and Labels ({len(labels)} candidates to label) sheets of {WORKBOOK_PATH}")
    print(labels.groupby("query_id").size().describe()[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
