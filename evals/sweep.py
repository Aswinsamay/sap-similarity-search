"""Search the whole catalogue for each test product's kind of product, and queue the matches for labelling.

Run once from the repo root, after evals/shortlist.py and its labels:

    python -m evals.sweep

The first round (evals/shortlist.py) read about 37 candidates per test product, chosen by four
computer routes. The ground truth could then only contain what those routes surfaced: for the
kalamkari blouse fabric, every route leaned on its wrong category (kurtis), so no fabric was
ever read.

This sweep fixes that. For each test product, the terms below describe its kind of product in
words taken from its own title (garment type, audience, a key detail) plus a price band around
its own price. They are never taken from any method's results. Every product among the 30,000
that matches is read and labelled. The terms were tightened until each sweep is about 150 to
200 products. Where fewer match, all of them are read.

New rows are appended to the Labels sheet of evals/eval_results.xlsx with `good` and `gt_rank`
empty and `source` set to "sweep". Rows from the first round have `source` "shortlist". A test
product that already has sweep rows is skipped, so running this twice adds nothing.
"""

from __future__ import annotations

import pandas as pd

from evals.store import WORKBOOK_PATH, load_data, save_data
from similarity.features import genders
from similarity.preprocessing import load_products


def sweep_masks(products: pd.DataFrame, queries: pd.DataFrame) -> dict[str, pd.Series]:
    """One boolean mask over all products per test product. The comment gives the count read."""
    name = products["name"].str.lower()
    price = products["price"]
    category = products["child_category"].fillna("")
    gender = genders(products)
    women, men, boys, girls = (gender == "women"), (gender == "men"), (gender == "boys"), (gender == "girls")

    def has(pattern: str) -> pd.Series:
        return name.str.contains(pattern, regex=True)

    def band(query_id: str, low: float, high: float) -> pd.Series:
        """Price between low x and high x the test product's price. Unknown prices are kept."""
        own = queries.loc[query_id, "price"]
        return price.isna() | ((price >= own * low) & (price <= own * high))

    kurta = has(r"\bkurt(?:a|i|is|as)\b")
    # As run, this also matches "...fit Shirt" (the "t" of "fit"), so some collared shirts were swept
    # into Q03, Q04, Q09, Q17 and Q20 and labelled not good, and "fit Shirt" titles were left out of
    # Q06 and Q08. Kept unchanged so this file records exactly what was searched.
    tee = has(r"t-?\s?shirt|\btees?\b|tshirt")
    shirt = has(r"shirt") & ~tee
    saree = has(r"saree|\bsari\b")
    fabric = has(r"fabric|running material|blouse material|kurti material|unstitched") | (category == "WomensEthnicUnstitchedFabric")
    characters = has(r"kidsville|marvel|batman|superman|captain|spider|avengers|disney|iron ?man|hulk")

    return {
        # kurta set with dupatta, 0.5x-2x price: 127 new
        "Q01": women & kurta & has(r"\bset\b|dupatta|palazzo|salwar|sharara|with pants?") & band("Q01", 0.5, 2),
        # block-print or cotton/printed anarkali kurti, 0.5x-2x: 134
        "Q02": women & kurta & (has(r"block ?print") | (has(r"anarkali") & has(r"cotton|print"))) & band("Q02", 0.5, 2),
        # men's plain/solid regular-fit T-shirt, 0.75x-1.33x: 181
        "Q03": men & tee & has(r"plain|solid") & has(r"regular") & band("Q03", 0.75, 1.33),
        # men's plain/solid slim-fit T-shirt, 0.5x-2x: 163
        "Q04": men & tee & has(r"plain|solid") & has(r"slim") & band("Q04", 0.5, 2),
        # silk or satin saree, 0.4x-2.5x: 201
        "Q05": saree & has(r"silk|satin") & band("Q05", 0.4, 2.5),
        # men's linen shirt, or plain regular-fit formal shirt, 0.5x-2x: 194
        "Q06": men & shirt & (has(r"linen") | (has(r"formal") & has(r"plain|solid") & has(r"regular"))) & band("Q06", 0.5, 2),
        # girls' cotton frock or dress, 0.5x-2x: 193
        "Q07": girls & has(r"frock|dress") & has(r"cotton") & band("Q07", 0.5, 2),
        # men's printed casual shirt, 1/3x-3x: 134
        "Q08": men & shirt & has(r"casual") & has(r"print") & band("Q08", 1 / 3, 3),
        # boys' regular-fit T-shirt, 0.85x-1.18x: 175
        "Q09": boys & tee & has(r"regular fit") & band("Q09", 0.85, 1.18),
        # women's a-line dress, 0.67x-1.5x: 197
        "Q10": women & has(r"dress") & has(r"a-?\s?line") & band("Q10", 0.67, 1.5),
        # newsboy / flat / ivy / cabbie-style caps, not women's or girls': 57
        "Q11": ~women & ~girls & has(r"newsboy|flat ?cap|\bivy\b|cabbie|gatsby|beret|driver cap|duckbill"),
        # men's cotton briefs (not boxer briefs or trunks), packs or poco: 196
        "Q12": men & has(r"\bbriefs?\b") & ~has(r"boxer|trunk") & has(r"cotton") & has(r"pack|poco|jockey"),
        # women's fabrics by the piece: not sarees, suit sets or men's shirting; plus kalamkari / block-print dress materials, 1/3x-3x: 64
        "Q13": (
            (fabric & ~has(r"saree|\bsari\b|dress material|\bsuits?\b|salwar|lehenga|trouser|shirting|suiting|\bmen\b|men's|boys?\b"))
            | (has(r"dress material") & has(r"kalamkari|block|hand ?print"))
        ) & ~men & ~boys & band("Q13", 1 / 3, 3),
        # baby / toddler frock or dress, not boys', 0.5x-2x: 176
        "Q14": ~boys & has(r"frock|dress") & has(r"baby|infant|month|toddler|new ?born") & band("Q14", 0.5, 2),
        # women's straight kurta, cotton/rayon/Rajasthani, 0.75x-1.33x: 159
        "Q15": women & kurta & has(r"straight") & has(r"cotton|rajasthan|jaipur|rayon") & band("Q15", 0.75, 1.33),
        # cuffed / knit / logo beanies (no price known): 156
        "Q16": has(r"beanie|skull ?cap|knit cap|knit hat") & has(r"cuff|knit|logo|marvel|avengers|embroider"),
        # toddler / baby T-shirts, not girls' (no price known): 100
        "Q17": ~girls & tee & has(r"toddler|\b[2-5]t\b|infant|baby|little boys?"),
        # women's plain / solid / orange / shift dress or frock, not kurtas (no price known): 101
        "Q18": women & has(r"dress|frock|frok|gown") & ~kurta & has(r"plain|solid|orange|shift"),
        # every OOMPH! saree, plus silk sarees at 0.92x-1.08x: 209
        "Q19": saree & (has(r"oomph") | (has(r"silk") & band("Q19", 0.92, 1.08))),
        # boys' character T-shirts, or plain regular-fit boys' T-shirts at 0.75x-1.33x; all within 0.5x-2x: 188
        "Q20": boys & tee & (characters | (has(r"plain") & has(r"regular") & band("Q20", 0.75, 1.33))) & band("Q20", 0.5, 2),
    }


def main() -> None:
    products = load_products()
    gender = genders(products)
    all_queries, labels = load_data()
    queries = all_queries.set_index("query_id")

    masks = sweep_masks(products, queries)
    new_rows = []
    for query_id, mask in masks.items():
        mine = labels[labels["query_id"] == query_id]
        if (mine["source"] == "sweep").any():
            print(f"{query_id}: already swept, skipped")
            continue
        own_asin = products.loc[products["uniq_id"] == queries.loc[query_id, "uniq_id"], "asin"].iloc[0]
        pool = products[mask & (products["asin"] != own_asin)].drop_duplicates("asin")
        pool = pool[~pool["uniq_id"].isin(mine["uniq_id"])]
        start = int(mine["position"].max())
        for position, (row, item) in enumerate(pool.iterrows(), start=start + 1):
            new_rows.append({
                "query_id": query_id, "position": position, "uniq_id": item["uniq_id"], "name": item["name"],
                "category": item["child_category"], "gender": gender.loc[row], "price": item["price"],
                "brand": item["brand"], "colours": "|".join(item["colours"]) if item["colours"] else None,
                "good": None, "gt_rank": None, "source": "sweep",
            })
        print(f"{query_id}: {len(pool)} new products to label")

    if new_rows:
        labels = pd.concat([labels, pd.DataFrame(new_rows)], ignore_index=True)
        save_data(all_queries, labels)
    print(f"{len(new_rows)} rows added; the Labels sheet of {WORKBOOK_PATH} now has {len(labels)} rows")


if __name__ == "__main__":
    main()
