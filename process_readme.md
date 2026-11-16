# Process: product similarity search

A running record of what each stage found, what it does, and why. It is updated as each stage is finished.

Data: `data/archive/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`

Stages 1 to 3 each have a module in `similarity/` that holds the logic and a numbered notebook at the repo root that walks through it. Notebooks call the module's functions; they never copy its logic. Stage 4 adds the web layer (`app.py`, `Dockerfile`) and stage 4b the Kubernetes files (`k8s/`). The evaluation (stage 5) is kept apart in `evals/`. Run everything from the repo root.

## Pipeline

```
raw ldjson ─► 01 preprocessing ─► data/processed/products.parquet ─► 02 features ─► 03 similarity search ─► 04 API + Docker ─► 4b Kubernetes
                                                                                 ├─► 05 evaluation (scores every method)
                                                                                 ├─► 06 text improvements
                                                                                 ├─► 07 ANN index (bonus)
                                                                                 └─► 08 images (optional)
```

| Stage | Module | Notebook | Output | Status |
|---|---|---|---|---|
| 0 EDA | — | `eda.ipynb`, `eda_learning.ipynb` | Findings below | Done |
| 1 Preprocessing | `similarity/preprocessing.py` | `01_preprocessing.ipynb` | `data/processed/products.parquet` | Done |
| 2 Features | `similarity/features.py` | `02_features.ipynb` | `Features` object, built in memory | Done |
| 3 Similarity search | `similarity/search.py` | `03_search.ipynb` | `find_similar_products` | Done |
| 4 API + Docker | `app.py`, `Dockerfile` | `04_api.ipynb` | `GET /find_similar_products`, Docker image | Done |
| 4b Kubernetes (optional) | `k8s/` | — (terminal commands below) | Deployment + Service on Docker Desktop | Done |
| 5 Evaluation | `evals/` | — (`python -m evals.run_eval` writes the workbook) | `evals/eval_results.xlsx` | Done |
| 6 Title embeddings | `similarity/embeddings.py` | `06_text_embeddings.ipynb` | `data/processed/title_embeddings.npz`, method `minilm` | Next |
| 7 Images (optional) | `similarity/images.py` | `07_image_similarity.ipynb` | Photos (outside the repo), `data/processed/image_embeddings.npz`, methods `tfidf_image`, `minilm_image` | Planned |
| 8 ANN index (bonus) | — | — | HNSW index, recall@k | Planned |

When each part runs:

| When | What | Time |
|---|---|---|
| Once, when the Docker image is built | Stage 1 parses the raw file into `products.parquet` | about 3 s |
| Once, when the server starts | Stage 2 turns the table into arrays: TF-IDF matrix, codes, numbers | about 0.5 s (2.8 s with imports) |
| A request for a new product | Stage 3 scores it against all 30,000 products; the best 100 are kept | about 22 ms in Python, 38 ms over HTTP |
| A request for a product asked before | The kept 100 are cut to `num_similar` | about 2 ms over HTTP |

---

## Stage 0: EDA

`eda.ipynb` is the raw pass and reproduces every number in this section. `eda_learning.ipynb` covers the original cells with a note on each; it does not have the later cells for duplicates, reviews, weight errors and offer fields. The only charts, two price histograms, are drawn in `eda.ipynb`.

### Key findings

1. **`uniq_id` is the only unique key, and some products appear twice.** 451 ASINs appear on 922 rows. In 88.5% of these groups, name, brand, price, rating, colour and weight are identical: the same listing was scraped twice.
2. **`product_name` carries most of the signal.** It is the only descriptive field filled on every row.
3. **Child category separates products; parent category does not.** 98% of parent categories are `ClothingAccessories`. `browsenode` covers 98% of rows and fills 5,050 of the 5,149 rows with no child category.
4. **Colour is sparse and is a list.** It is 80% missing, and each value lists the colours a listing is sold in (`Black|White`).
5. **Weight is mostly a placeholder.** 79% of rows hold the sentinel `999999999`, and some real values have unit errors.
6. **Price is right-skewed.** Median ₹590, max ₹9,988, so it is compared on a log scale.
7. **Rating mostly reflects how few reviews a product has.** 50% of single-review products are rated exactly 5.0; none with more than 20 reviews are.
8. **Review counts sit in two fields that almost never overlap.** Where both are present, they usually disagree.
9. **Prime, discount, delivery, stock and seller describe the offer, not the product.** Prime flips between the two scrapes of the same product in 14% of duplicate pairs.

### Dataset

| | |
|---|---|
| Rows | 30,000 |
| Columns | 33 |
| Distinct `uniq_id` | 30,000 |
| Distinct `asin` | 29,529 |
| Repeated `product_name` | 7,576 |
| Crawl window | 6 Feb 2020 02:02 to 7 Feb 2020 14:51 UTC (37 hours, despite the filename) |

### Column inventory

"Filled" counts non-empty, non-placeholder values out of 30,000.

| Group | Column | Contents | Filled | Role |
|---|---|---|---:|---|
| Identity | `uniq_id` | Row hash made by the scraper | 100% | **Key** |
| | `asin` | Amazon product ID | 100% | De-duplication |
| | `product_url` | amazon.in page | 100% | Not used |
| | `crawl_timestamp` | Scrape time | 100% | Not used |
| Text | `product_name` | Listing title | 100% | **Feature** |
| | `meta_keywords` | Title, often with brand or manufacturer appended | 100% | Not used (redundant) |
| Attributes | `brand` | Brand name; `Generic` is a placeholder | 72% | **Feature** |
| | `colour` | Pipe-separated colour list | 20% | **Feature** |
| | `sales_price` | Price in rupees | 90% | **Feature** |
| | `weight` | Text weight; `999999999` is the sentinel | 21% | **Feature** |
| | `rating` | Average stars, 1 to 5 | 100% | Tie-break |
| Category | `sales_rank_in_child_category` | `{category: "#rank"}` | 83% | **Feature** (key only) |
| | `browsenode` | Amazon category-tree node ID | 98% | Category fallback |
| | `sales_rank_in_parent_category` | `{category: "#rank"}` | 85% | Not used |
| | `parent___child_category__all` | Parent and child dictionaries merged | 85% | Not used (redundant) |
| Images | `medium` | Base-size image URLs, `\|`-separated | 100% | **Image feature** |
| | `large` | 1500 px image URLs | 96% | Not used |
| | `image_urls__small` | 38×50 px thumbnails | 100% | Not used |
| Nested | `product_details__k_v_pairs` | Reviews text, ASIN, first-available date, bestseller rank, part number, weight, dimensions | 96% | Review count |
| | `technical_details__k_v_pairs` | Brand, colour, material, size | 4% | Not used (too sparse) |
| Offer | `discount_percentage` | Markdown such as `40%` | 49% | Not used |
| | `amazon_prime__y_or_n` | Y / N | 100% | Not used |
| | `best_seller_tag__y_or_n` | Y on 13 rows | 100% | Not used |
| | `delivery_type` | Fulfilled by merchant or by Amazon | 100% | Not used |
| | `left_in_stock` | "Only N left", values 1 to 5 | 10% | Not used |
| | `no__of_offers`, `no__of_sellers` | Equal on all 1,020 rows | 3% | Not used |
| | `seller_name`, `seller_id` | 2,333 sellers; the name matches the brand on 1,582 rows | 28% | Not used |
| Behaviour | `no__of_reviews` | Review count | 12% | Review count |
| | `other_items_customers_buy` | About 9 titles bought together, `\|`-separated | 81% | Not used |
| Books | `formats___editions`, `name_of_author_for_books` | Left over from a books schema | 3 rows | Not used |

### Identity and duplicates

- **ASIN** (Amazon Standard Identification Number) is Amazon's 10-character product ID. There is one per product, not per seller; size and colour variants usually get their own.
- `uniq_id` is the scraper's hash, so one page scraped twice yields two `uniq_id`s with one ASIN.
- 451 ASINs appear on 922 rows, 88.5% of them identical on the key attributes. Without handling, a search returns the product's own twin as result #1.
- Repeated titles are a different case. Rows whose title appears more than once cover 9,676 distinct ASINs, mostly size or colour variants. These are legitimately similar.
- The ID inside `product_url` matches `asin` on only 11,328 rows, but maps one-to-one to it, so it adds nothing.

### Price

`sales_price` loads as a number for 27,110 rows (90.4%). Nothing is zero or negative. `eda.ipynb` draws a histogram of the price and one of its log10, which shows the skew.

| Min | 5th pct | 25th pct | Median | 75th pct | 95th pct | 99th pct | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| ₹39 | ₹219 | ₹379 | ₹590 | ₹899 | ₹2,877 | ₹5,220 | ₹9,988 |

There is no separate `price` column, even though the README lists both "price" and "sales_price".

### Rating and reviews

Rating is present on all 30,000 rows; 3,861 are exactly 4.0 and 7,450 exactly 5.0.

**Rating against review count** (count from `product_details`, 22,205 rows):

| Reviews | Products | Mean rating | Std | Exactly 5.0 |
|---|---:|---:|---:|---:|
| 1 | 8,065 | 4.32 | 0.87 | 50% |
| 2 | 3,353 | 4.11 | 0.75 | 23% |
| 3–5 | 4,416 | 3.92 | 0.63 | 8% |
| 6–20 | 4,189 | 3.75 | 0.47 | 1% |
| 21–100 | 1,735 | 3.74 | 0.36 | 0% |
| 100+ | 447 | 3.79 | 0.32 | 0% |

With few reviews, ratings sit at the extremes; with many, they converge on about 3.75. A 5.0 mostly signals a small sample.

**Two sources for the review count:**

| Where the count is found | Rows |
|---|---:|
| `product_details` text only (`"3 customer reviews"`) | 22,139 |
| `no__of_reviews` column only | 3,386 |
| Both (agree on 5) | 66 |
| Neither | 4,409 |

Where the column is filled, the details text usually reads *"Be the first to review this item"*. The column's median is 15 reviews and the text's is 2; the two likely come from different page widgets, but the data cannot confirm it. 6,611 rows say *"Be the first to review this item"* yet carry a rating.

### Weight

| | Rows | Percent |
|---|---:|---:|
| Sentinel `999999999` | 23,745 | 79.1% |
| Parsed to grams | 6,254 | 20.8% |
| Unparsed text | 1 | — |

- Units seen: `g` (6,113), `Grams`/`grams` (120), `Kg`/`Kilograms` (20), and one value with a garbled byte before `grams`.
- 8 Truform compression stockings say `113 Kilograms` or `136 Kilograms`: grams with the wrong unit.
- 62 products weigh exactly `4.54 g`, which is 0.01 lb: a placeholder.
- Without those, weights run from 4.54 g to 4,540 g, median 240 g.
- `Item_Weight` in `product_details` (3,821 rows) adds no weights that `weight` lacks.

### Brand and colour

| | Present | Distinct values |
|---|---:|---:|
| Brand | 21,857 | 6,458 |
| Colour | 6,029 | 4,757 |
| Both | 4,791 | — |

- Largest brands: Max (504), Generic (245), BIBA (205), Mothercare (156), Campus Sutra (150). `Generic` (245) and `N/A` (4) are placeholders.
- The seller is not the brand: `seller_name` matches `brand` on only 1,582 rows.
- Every non-empty `colour` lists at least two colours, e.g. `Black|White` (79). Raw string matching almost never hits, so colours are compared as sets.
- 8,087 more rows have no `colour` but a colour word in `product_name` (35-word list in `eda.ipynb`). A vocabulary built from the `colour` column does not work: it picks up junk such as `as`, `april` and `b`.

### Category

| | Present | Distinct values |
|---|---:|---:|
| `browsenode` | 29,480 (98.3%) | 302 |
| Child category | 24,851 (82.8%) | 218 |
| Parent category | 25,497 (85.0%) | 19 |

- Parent category is `ClothingAccessories` for 24,893 of 25,497 rows.
- Child category is the key of `sales_rank_in_child_category`; the rank measures popularity, not what the product is.
- `browsenode` is finer: each child category spans 2.7 nodes on average, up to 20.
- Largest child categories: WomensKurtasKurtis 3,534, MensT_Shirts 3,385, WomensSarees 1,808, MensFormalShirts 1,060, GirlsDressesJumpsuits 992.

### Text and images

- `product_name`: min 4, median 50, max 500 characters. It packs in brand, garment type, material, fit and often colour.
- `meta_keywords` equals the title on only 6.4% of rows; otherwise it is the title plus tokens such as `,LOVISTA`.
- `other_items_customers_buy` lists about 9 bought-together titles per row (up to 114). These are complementary items, not substitutes.

| Image field | Present | Size (12-image sample) | Median file |
|---|---:|---|---:|
| `image_urls__small` | 29,998 | 38×50 px | 0.9 KB |
| `medium` | 29,998 | ~500 px long side | 28 KB |
| `large` | 28,841 | ~1500 px long side | 141 KB |

Each field packs several URLs separated by `|`; the first is the main photo. `medium` is the smallest size above the 224 px a CNN takes, so nothing useful is lost and it is 5× cheaper to download than `large`. All 12 sampled URLs still loaded.

### Offer fields

| Field | Values |
|---|---|
| `amazon_prime__y_or_n` | N 16,500, Y 13,500 |
| `best_seller_tag__y_or_n` | N 29,987, Y 13 |
| `delivery_type` | Fulfilled by merchant 26,382, by Amazon 3,618 |
| `discount_percentage` | 14,624 parsed, median 54, range 1 to 99 |

- Prime is not a product property: it differs between the two scrapes of the same ASIN in 64 of 451 duplicate groups.
- Prime is not fulfilment by Amazon: 11,327 merchant-fulfilled rows are Prime.
- Prime mostly encodes category: 23% of boys' T-shirts against 86% of men's shirts.

These fields are candidates for filters or business re-ranking ("Prime only"), not for the similarity score.

---

## Stage 1: Preprocessing

| | |
|---|---|
| Module | `similarity/preprocessing.py` |
| Notebook | `01_preprocessing.ipynb` |
| Tests | `tests/test_preprocessing.py` (41 tests) |
| Run | `python -m similarity.preprocessing` (about 3 s) |
| Output | `data/processed/products.parquet`: 30,000 rows, 13 columns, 3.3 MB |

### Method

1. `load_raw` reads only the 13 raw columns the project uses.
2. One parser per field: one raw value in, one clean value out, or `None` when the value is missing or unusable.
3. `build_products` runs every parser over every row and adds `rating_adjusted`.
4. `validate` raises an error listing every failed check: unique non-empty `uniq_id`, a name on every row, positive prices, weights in (0, 20,000] g, ratings and adjusted ratings in [1, 5], no negative review counts.
5. `save` writes Parquet. `load_products` reads it back for the later stages.

Running the module calls `main()`, which does steps 1 to 5. The notebook calls the same functions one at a time.

### Output schema

| Column | Type | Rule | Filled |
|---|---|---|---:|
| `uniq_id` | str | Unchanged | 30,000 |
| `asin` | str | Unchanged; used to drop duplicates at search time | 30,000 |
| `name` | str | Whitespace collapsed; lowercasing left to the text stage | 30,000 |
| `brand` | str | Lowercased, trimmed; `generic` and `n/a` → null | 21,608 (72.0%) |
| `colours` | list[str] | Split on `\|`, lowercased, de-duplicated, sorted | 6,029 (20.1%) |
| `price` | float | `sales_price`, kept only if > 0 | 27,110 (90.4%) |
| `weight_g` | float | Units → grams; sentinel, sentences, > 20 kg and 4.54 g → null | 6,184 (20.6%) |
| `rating` | float | Kept if in [1, 5] | 30,000 |
| `review_count` | float | Details text, then `no__of_reviews`, then "Be the first" = 0 | 29,483 (98.3%) |
| `child_category` | str | Key of the rank dictionary | 24,851 (82.8%) |
| `browsenode` | str | Node ID stored as text, not float | 29,480 (98.3%) |
| `image_url` | str | First `medium` URL | 29,998 (100%) |
| `rating_adjusted` | float | Bayesian average, see below | 30,000 |

Where `review_count` comes from: details text 22,205, `no__of_reviews` column 3,386, "Be the first" = 0 for 3,892, unknown 517.

### Adjusted rating

```
rating_adjusted = (C·m + n·r) / (C + n)      C = 5, m = 4.18
5.0 from 1 review    → 4.31
4.4 from 200 reviews → 4.39
unknown review count → 4.18
```

`m` is the average star over every review in the dataset. Because it is weighted by review count, heavily reviewed products pull it above the 3.75 that well-reviewed products average per product. After adjustment, the top 10 are all 5.0 ratings backed by 246 to 1,154 reviews.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Parquet output | Keeps types, nulls and lists; acts as a typed contract between stages; loads with no parsing; readable by pandas, Polars, DuckDB | Binary, not diffable; adds `pyarrow` |
| `pyarrow==25.0.1` | 26.x requires Python 3.11, and the Dockerfile uses 3.10 | One major version behind |
| Keep all 30,000 rows | Every `uniq_id` in the dataset stays valid for the API | Stage 3 must drop the query's twin and collapse twins in results |
| Null instead of imputing | A guessed value makes unrelated products look similar | Fewer attributes to compare per pair |
| Review count from text first | Covers 22k rows and describes this exact listing | The two sources disagree; this is a judgment call |
| Parse here, compare later | `child_category` and `browsenode` stay separate columns; title colours are not merged into `colours` | Stage 2 has to apply the fallback |
| Output not in git | Regenerated from raw data + code, so it can never go stale; no binary bloat in history | One command to run before notebooks 02+ |
| pandas only offline | Parsing happens once; the request path will use NumPy arrays | — |

### Open issues

- The colour values are noisy: `gray` and `grey`, and typos such as `blackk`, appear as separate colours. Stage 2 fixes the common variants with a 6-entry map.
- ~~The Dockerfile needs `RUN python -m similarity.preprocessing` so the image builds its own data.~~ Done in stage 4: the Dockerfile's build stage runs it.

---

## Stage 2: Features

| | |
|---|---|
| Module | `similarity/features.py` |
| Notebook | `02_features.ipynb` |
| Tests | `tests/test_features.py` (20 tests) |
| Run | `python -m similarity.features` (about 0.6 s) |
| Output | A `Features` object in memory; row *i* of every array is product *i* |

### Method

| Feature | Built from | Technique | Result |
|---|---|---|---|
| `text` | `name` | Lowercase; every form of "t-shirt" → `tshirt`; TF-IDF over single words that appear in at least 2 titles | 30,000 × 7,412 matrix; each row has length 1, except 2 titles with no vector |
| `brand`, `category`, `browsenode`, `asin` | same columns | One integer per distinct value, -1 when missing | 6,336 brands, 218 categories, 302 nodes |
| `gender` | `child_category`, `name` | The category's first word (Mens, Womens, Maternity → women, Boys, Girls); otherwise the title, if it names exactly one gender. Then integer codes. | 27,346 known (91.2%): women 12,039, men 10,390, boys 2,731, girls 2,186 |
| `colours` | `colours` | 6 spelling fixes, then a 0/1 matrix with one column per colour | 30,000 × 7,819; 733 products changed by the fixes |
| `price`, `weight`, `rating` | `price`, `weight_g`, `rating_adjusted` | Kept as they are | NaN when missing |

### How TF-IDF works

A computer can only compare numbers, so each title becomes a list of word weights:

1. **Split** the title into words: "People Men's Printed Regular fit T-Shirt" → people, men, printed, regular, fit, tshirt.
2. **Score rarity (IDF)**, roughly `log(total titles ÷ titles containing the word)`. A word in many titles says little; a rare one says a lot.

   | men | shirt | printed | kurti | people | gota |
   |---:|---:|---:|---:|---:|---:|
   | 8,797 titles | 8,130 | 3,064 | 1,287 | 177 | 14 |
   | IDF 2.23 | 2.31 | 3.28 | 4.15 | 6.13 | 8.60 |

3. **Weight** each word by term frequency × IDF, then rescale so every title has the same total length.
4. **Compare** two titles with cosine similarity: multiply the weights of the words they share and add them up. 0 = no shared words, 1 = identical.

### Example

| | Category | Price | Text score vs A |
|---|---|---:|---:|
| **A** People Men's Printed Regular fit T-Shirt | MensT_Shirts | ₹195 | — |
| **B** Sf Jeans By Pantaloons Men's Plain Slim fit T-Shirt | MensT_Shirts | ₹265 | 0.15 |
| **C** Pepe Jeans Men's Solid Regular fit Casual Shirt | MensCasualShirts | ₹919 | 0.22 |
| **D** LOVISTA … Printed Kurti with Pant … | WomensKurtasKurtis | ₹660 | 0.06 |

Text alone ranks the casual shirt (C) above the other T-shirt (B), because A and C share "regular fit". Category and price both favour B, which is why stage 3 combines attributes. Searching on text alone, A's 10 closest titles are all *People* men's regular-fit T-shirts (scores 1.00 and 0.93): TF-IDF works well when the wording matches.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Word TF-IDF as the baseline | Builds in 0.6 s, needs no download, every match can be explained, rare words such as `gota` or a brand count strongly | No synonyms: a frock and a dress from the same category score 0.0, as do a kurta and a kurti. Exact spelling only. Word order is ignored. |
| No stop-word list | scikit-learn's English list removes `top`, `bottom`, `front`, `back`, `full`, `off`, `under`, which are garment words | Common words stay, but IDF already gives them little weight |
| Words in at least 2 titles | A word in one title can never be shared; dropping them shrinks the vocabulary from 16,660 to 7,412 | 2 titles (`Bauer Uncategorized`, `SilverGrey-LadiesChuridar`) end up with no text vector |
| "t-shirt" → `tshirt` | The tokenizer drops the one-letter "t", so a T-shirt read as "shirt"; 5,910 titles are affected | A hand-written rule |
| No word pairs | In the test they lowered every score without changing the ranking | Phrases like "slim fit" are not one unit |
| Integer codes | A code is only a label, compared with `==`, never subtracted. Comparing 30,000 codes takes 0.008 ms against 0.39 ms for 30,000 strings. Equality gives the same score as cosine on a one-hot vector. | -1 means missing, and -1 == -1 is True, so stage 3 must skip a pair when either side is -1 |
| 6-entry colour map | Fixes the most common spelling variants (`gray`/`grey`, `navy`/`navy blue`) | Other variants remain |
| Gender from the category first, the title second | Every category belongs to exactly one gender. Where both give a gender, the title agrees with the category 99.1% of the time (20,460 products). The title fills 2,555 products with no category. | 2,654 stay unknown, among them titles naming two genders ("for Men and Women") or "unisex", and titles with no gender word ("Cotton Saree"). Gender adds nothing for two products in the same category; it gives partial credit across categories (a men's polo vs a men's T-shirt). |
| Built in memory, not saved | 0.6 s to build, so a saved copy would only go stale | Rebuilt at every start |

### Open issues

- Colour variants not covered by the map: `multi-colored`, `multi coloured`, `multi-coloured`, `blackk`.
- The vocabulary contains model codes such as `011h8b9627`. A code shared by two titles usually links two variants of one product, which helps.
- The 2 titles with no text vector score 0 on text in stage 3. They were first treated as "text missing", which let one score a perfect 1.0 on browsenode alone (see stage 3).

---

## Stage 3: Similarity search

| | |
|---|---|
| Module | `similarity/search.py` |
| Notebook | `03_search.ipynb` |
| Tests | `tests/test_search.py` (14 tests; 75 across all stages) |
| Run | `python -m similarity.search <uniq_id> [num_similar]` |
| Output | `find_similar_products(product_id, num_similar=10) -> list[str]`, about 22 ms per query |

### Method

1. **Find the query's row** from its `uniq_id`.
2. **Score every product on each attribute**, from 0 to 1. numpy compares the query with all 30,000 products in one operation, without a Python loop.
3. **Combine** the scores into a weighted average over the attributes both products have.
4. **Remove duplicates**: the query, every row with the query's ASIN, and all but the best row of any other ASIN.
5. **Sort** by score rounded to 3 decimals, then adjusted rating, then `uniq_id`, and return the first N.

| Attribute | Weight | Score | Missing when |
|---|---:|---|---|
| Text | 0.30 | Cosine of the TF-IDF rows | Never: every product has a title |
| Category | 0.20 | 1 if same child category; where a pair cannot compare categories, 1 if same browsenode | Neither can be compared |
| Gender | 0.10 | 1 if same | Either is unknown |
| Price | 0.15 | Smaller ÷ larger: ₹500 vs ₹1,000 and ₹5,000 vs ₹10,000 both score 0.5 | Either is missing |
| Brand | 0.15 | 1 if same | Either is missing |
| Colour | 0.05 | Jaccard: colours in common ÷ colours of either | Either has none |
| Weight | 0.05 | Smaller ÷ larger | Either is missing |

There are no labelled "similar" pairs in the data, so the weights are a reasoned judgement. They follow three questions: does the attribute define what the product is, how often is it present, and how reliable is it. Brand is moderate because one brand spans many garments ("Max" sells in 28 categories) and is often in the title already. The weights can be overridden per call.

### How one search works, step by step

Think of the 30,000 products as rows of a spreadsheet. Stage 2 already turned every attribute into a column of numbers. A search adds new columns, each filled for all 30,000 rows in one numpy operation, then sorts the sheet.

**Query: A, "People Men's Printed Regular fit T-Shirt".** MensT_Shirts, men, ₹195. It has no brand, no colour and no weight.

**Step 1. Find the row.** `row_of` maps the `uniq_id` to row 3.

**Step 2. Score every attribute.** Each scoring function returns two columns: the score (0 to 1) and whether the pair could be compared. Five of the 30,000 rows (– = skipped):

| Candidate | Text 0.30 | Category 0.20 | Gender 0.10 | Price 0.15 | Brand 0.15 | Colour 0.05 | Weight 0.05 |
|---|---:|---:|---:|---:|:-:|:-:|:-:|
| **P** People Men's Printed Regular fit T-Shirt, ₹199 | 1.00 | 1 | 1 | 0.98 | – | – | – |
| **K** Knightlook Men's Printed Cotton T-Shirt, no category, no price | 0.48 | 1 (browsenode) | 1 | – | – | – | – |
| **B** Sf Jeans … Men's Plain Slim fit T-Shirt, ₹265 | 0.15 | 1 | 1 | 0.74 | – | – | – |
| **C** Pepe Jeans Men's Solid Regular fit Casual Shirt, ₹919 | 0.22 | 0 | 1 | 0.21 | – | – | – |
| **D** LOVISTA … Printed Kurti with Pant, ₹660 | 0.06 | 0 | 0 | 0.30 | – | – | – |

- Brand, colour and weight are skipped for **every** candidate, because A has none of them.
- K has no child category, so the category column compares browsenodes instead: both are on node 1968123031, so it scores 1.
- K has no price, so price is skipped for K only.

**Step 3. Combine.** Add up weight × score over the attributes the pair has, then divide by those weights:

| | Sum of weight × score | Weights used | Score |
|---|---|---:|---:|
| P | 0.30×1.00 + 0.20 + 0.10 + 0.15×0.98 = 0.747 | 0.75 | **0.996** |
| K | 0.30×0.48 + 0.20 + 0.10 = 0.445 | 0.60 | **0.742** |
| B | 0.30×0.15 + 0.20 + 0.10 + 0.15×0.74 = 0.455 | 0.75 | **0.607** |
| C | 0.30×0.22 + 0 + 0.10 + 0.15×0.21 = 0.198 | 0.75 | **0.264** |
| D | 0.30×0.06 + 0 + 0 + 0.15×0.30 = 0.061 | 0.75 | **0.082** |

K is divided by 0.60, not 0.75, because its price is unknown. Its score reflects only what is known about it.

**Step 4. Remove duplicates.** Drop A itself and every row with A's ASIN (here only A). Among the rest, keep the best row of each ASIN.

**Step 5. Sort and cut.** Sort by score rounded to 3 decimals, then adjusted rating, then `uniq_id`. P comes 1st, K 57th, B 959th, C 5,303rd, D 18,084th. B is a fair match, but 3,271 products score above 0.5, mostly men's T-shirts that share more words with A. `find_similar_products(A, 5)` returns the `uniq_id`s of the first 5 rows.

| Step | Code |
|---|---|
| 1 | `features.row_of[product_id]` in `find_similar_products` |
| 2 | `attribute_scores`, which calls `text_scores`, `category_scores`, `code_scores`, `ratio_scores` and `colour_scores` |
| 3 | `combine` |
| 4–5 | `rank`: `np.lexsort` sorts by the three keys; `np.unique` finds the first row of each ASIN |
| Why a row scored what it did | `score_breakdown(features, a, b)` |

### Missing attributes

```
score = Σ weight × similarity   over the attributes both products have
        ─────────────────────
        Σ weight                 over the same attributes
```

A missing attribute is left out for that pair and the remaining weights are rescaled, so the score stays between 0 and 1. This is Gower's similarity coefficient (J. C. Gower, "A general coefficient of similarity and some of its properties", *Biometrics* 27(4), 1971), built for records that mix text-like, categorical and numeric attributes with gaps. It is also the README's "fallback mechanism when certain features are unavailable", and stage 8 images will use it too.

Missing data is the normal case, not the exception:

| Attributes present | 2 | 3 | 4 | 5 | 6 | All 7 |
|---|---:|---:|---:|---:|---:|---:|
| Products | 16 | 869 | 8,772 | 12,554 | 6,842 | 947 (3%) |

| Who lacks the attribute | Effect | Example above |
|---|---|---|
| The query | Skipped for every candidate, so it does not change the order | A has no brand: brand plays no part in A's results |
| One candidate | Skipped for that pair only | K has no price |
| Both | Skipped; two unknowns are never a match | — |
| Child category on either side | Browsenodes are compared instead, if both have one | K |

| Alternative | Why not |
|---|---|
| Missing counts as 0 | Unknown is treated as "different". Colour is missing for 80% of products, so nearly everyone would be penalised. |
| Fill in a typical value (median price) | Two unknown prices would both become ₹590, a perfect but invented match |
| Two missing codes count as a match | `-1 == -1` is True in numpy; the code checks for -1 on either side first |

### Duplicates and ties

- **ASIN.** Each `uniq_id` is a valid query, so all rows are kept and duplicates are handled at search time. For "Captain America By Kidsville Boy's Plain Regular fit T-Shirt", one ASIN on 3 rows, a search from the first row returns neither of the other two. Results hold one row per ASIN: asking for every product returns 29,528, the 29,529 ASINs minus the query's.
- **Ties.** Scores are rounded to 3 decimals, so near-equal scores tie and the higher adjusted rating comes first. For A, four "People Men's Regular fit T-Shirt" listings score 0.967 and are ordered by rating 4.23, 4.18, 4.15, 3.98. `uniq_id` breaks any remaining tie, so the same request always returns the same list.

### Results

A ("People Men's Printed Regular fit T-Shirt", ₹195): the top 10 are all *People* men's regular-fit T-shirts at ₹179 to ₹219, scoring 0.996 to 0.949.

Combining attributes fixes the stage 2 example, where text alone ranked the casual shirt C above the T-shirt B:

| vs A | Text only | Combined |
|---|---:|---:|
| **B** Men's Plain Slim fit T-Shirt, MensT_Shirts, ₹265 | 0.149 | **0.607** |
| **C** Men's Solid Regular fit Casual Shirt, MensCasualShirts, ₹919 | 0.220 | 0.264 |
| **D** Printed Kurti with Pant, WomensKurtasKurtis, ₹660 | 0.057 | 0.082 |

Checks without labels, averaged over the top 10 of 500 random products (`result_checks`):

| | Text only | Combined |
|---|---:|---:|
| Same category as the query | 67.8% | 97.6% |
| Same gender | 87.6% | 99.7% |
| Median price ratio | 0.73 | 0.87 |
| Shares no title word with the query | 0% | 1.9% |
| Evidence weight, top 10 | 0.82 | 0.79 |
| Evidence weight, all products | 0.79 | 0.79 |

- Category and gender are inputs to the combined score, so their rows show that these attributes take effect, not an independent quality measure. Reading examples in the notebook is the main check.
- Evidence weight is the total weight a pair was judged on. The top 10 are judged on as much as an average product, so the skip-and-rescale rule is not pushing products with gaps upward. A neutral 0.5 for missing attributes was the planned fallback; it is not needed.

**Does gender earn its place, given that the category name already contains it?** Same 500 queries, with and without the gender attribute:

| | With gender | Without |
|---|---:|---:|
| Same gender as the query | 99.7% | 98.3% |
| Same category | 97.6% | 97.2% |
| Queries with at least one wrong-gender result in the top 10 | — | 32 of 454 |
| Queries whose top 10 changed | 121 of 500 (about 0.4 results each) | |

- Category scores all-or-nothing, so it cannot tell "different category, same gender" from "different category, different gender". Gender can.
- Example: without gender, "United Colors of Benetton Boy's Sweatshirt" gets two Benetton *men's* sweatshirts at 9th and 10th. They match on brand, price and title words, and category scores 0 either way.
- Gender also helps queries with no category: their top 10 changed in 35% of cases, against 22% for queries with one, because the title still gives them a gender.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| A separate score per attribute, combined by weights | Each attribute's influence can be set and explained; `score_breakdown` shows every attribute's part of a score | The weights are judgement, not learned |
| Gender as its own attribute | Partial credit for the same gender in another category | Two products in the same category always share gender, so an exact category match is worth 0.30, not 0.20. This gives three levels: same category > same gender only > neither. |
| Skip and rescale for missing attributes | Judges each pair on what is known; only 3% of products have every attribute | A pair with gaps is judged on less evidence; measured above and not a problem on average |
| Text is never missing | First version treated the 2 empty text vectors as missing. "Bauer Uncategorized" then scored a perfect 1.0 against a Marvel beanie on browsenode alone and ranked #1. An empty vector means no shared words, so 0 is the true score. Every pair is now judged on at least text. | None found |
| Category falls back to browsenode | Leaves 99 products with neither, instead of 5,149 without a category. Two products sharing a browsenode share the category 96.9% of the time. | Some categories span many browsenodes (MensT_Shirts covers 17), so the fallback is slightly stricter |
| Ratio for price and weight | No unit, and treats ₹100 vs ₹200 like ₹1,000 vs ₹2,000 | Generous: ₹200 vs ₹250 still scores 0.8 |
| Scores rounded to 3 decimals before the tie-break | Differences below 0.001 are noise; the rating then decides | Rounding hides very small real differences |
| One row per ASIN in results | The same product never fills two slots | The other row is not shown; twins differ at most in price or offer |
| Exact search over every product | 22 ms per query at 30,000 products; the exact answer that stage 7's ANN index is measured against | Time grows with catalogue size |

### Open issues

- **Category, gender and price can outvote text.** Query: "LA' Facon Cotton Kalamkari Handblock Saree Blouse Fabric", filed under WomensKurtasKurtis at ₹200.
  - Kurtis at ₹200 that share no title word score 0.60 and fill the top 10.
  - Items sharing "kalamkari" or "handblock" score about 0.24 on text but differ in price or category. A handblock kurta at ₹543 ranks 540th; kalamkari sarees rank 3,552nd and lower.
  - 1.9% of top-10 results share no word with the query. Better text similarity (stage 6) or a higher text weight would help; both should be measured with the stage 5 evaluation before changing.
- The weights cannot be validated without labelled pairs.
- 2,654 products have no gender, and 99 have neither category nor browsenode.

---

## Stage 4: API and Docker

| | |
|---|---|
| Module | `app.py` at the repo root, as in the README's sample |
| Notebook | `04_api.ipynb` |
| Tests | `tests/test_app.py` (13 tests; 88 across all stages) |
| Run locally | `uvicorn app:app --port 8000`, then open `http://localhost:8000/docs` |
| Run in Docker | `docker build -t product-similarity .`, then `docker run --rm -p 8000:8000 product-similarity` |
| Image | 684 MB unpacked, 206 MB compressed (the download size) |
| Output | `GET /find_similar_products?product_id=…&num_similar=…` returns a JSON list of `uniq_id`s |

### Method

1. **Start up.** FastAPI's `lifespan` step builds the features once. uvicorn opens the port only after it finishes, so no request can reach a half-loaded server.
2. **Validate the request.** FastAPI reads the rules from the function signature. `product_id` must be present and not empty. `num_similar` must be a whole number from 1 to 100, and defaults to 10. A failed rule returns 422 before our code runs.
3. **Check the ID.** An unknown `product_id` returns 404. This check runs before the cache, so made-up IDs are never stored in it.
4. **Search, or reuse.** `top_similar(product_id)` runs stage 3 once for the best 100 and keeps the answer in a cache of 1,024 products. Any `num_similar` is a slice of those 100.
5. **Answer.** The slice, as a JSON list. Anything unexpected is written to the log with its traceback, and the caller gets a plain 500.

### Requests and answers

From `04_api.ipynb`, for A ("People Men's Printed Regular fit T-Shirt", the stage 3 example):

| Request | Status | Body |
|---|---|---|
| `product_id=25798d…&num_similar=5` | 200 | 5 `uniq_id`s: *People* men's T-shirts at ₹179 to ₹199 |
| `product_id=not-a-product` | 404 | `{"detail": "product_id not found: not-a-product"}` |
| No `product_id`, or an empty one | 422 | `{"detail": [{"type": "missing", "loc": ["query", "product_id"], …}]}` |
| `num_similar` = 0, 101 or `ten` | 422 | `{"detail": [{"type": "less_than_equal", "msg": "Input should be less than or equal to 100", …}]}` |
| A bug inside the search | 500 | `{"detail": "internal error while searching"}` |
| `GET /health` | 200 | `{"status": "ok", "products": 30000}` |

- **422 vs 404.** 422 means the request itself is wrong; it is FastAPI's standard code for a failed rule, and the body says which rule failed. 404 means a well-formed request for a product that does not exist. The caller can tell "fix your request" apart from "no such product".
- **`/health`** is for Docker and Kubernetes: it answers only once the data is loaded.
- **`/docs`** is **Swagger UI**, served by FastAPI: a page where each endpoint can be tried in the browser ("Try it out"), with the limits (1 to 100, default 10) shown. It is drawn from `/openapi.json`, a standard machine-readable description of the API that FastAPI builds from the function signature, so the page cannot drift from the code. `/redoc` shows the same as a read-only page. In production, `/docs` is often switched off or put behind a login.

### Who does what in a request

| Part | Job |
|---|---|
| **uvicorn** (the server) | Opens port 8000, reads the raw HTTP request, hands it to FastAPI, sends the answer back. FastAPI cannot listen on a port by itself. |
| **FastAPI** (the framework) | Matches the URL to a function, checks and converts the parameters (or returns 422), turns the returned list into JSON |
| **`app.py`** | The 404 check, the cache, the 500 handling |
| **`similarity/`** | The search itself (stage 3) |

uvicorn and FastAPI talk through ASGI, a standard interface between Python web servers and web apps. Compared with an older framework such as Flask, FastAPI does the input checking, JSON conversion and documentation from the type hints, where Flask code would read and check each parameter by hand. To a caller there is no difference: both are HTTP and JSON.

`uvicorn app:app --host 0.0.0.0 --port 8000` means: in `app.py`, serve the object named `app`, accept connections from any network (needed so traffic from outside the container gets in), on port 8000.

### The cache

- **Why always 100.** The cost of a search is scoring all 30,000 products, and that is the same whether 5 or 100 are kept. Keeping 100 lets `num_similar=3` and `num_similar=10` share one entry, and the 3 are always the first 3 of the 10.
- **Size.** At most 1,024 products, about 8 MB. When full, the product asked for least recently is dropped (LRU: least recently used).
- **Effect.** 200 random products in `04_api.ipynb`: 29.7 ms per request the first time, 5.6 ms the second. Over real HTTP: 38 ms and 2 ms.
- **No expiry needed.** The data is fixed inside the image. New data means a new image, which starts with an empty cache.
- **One cache per running copy.** With two containers, each keeps its own. That costs at most one extra 22 ms search per product per copy.

### The Docker image

The Dockerfile has two stages. Only what the server needs is copied into the second.

```
stage 1 "build" (python:3.10-slim)                stage 2, the image that runs (python:3.10-slim)
  venv + requirements.txt ───────────── copied ──►  /opt/venv
  similarity/ + raw ldjson (74 MB)
  python -m similarity.preprocessing
    → data/processed/products.parquet ── copied ──►  data/processed/products.parquet (3.3 MB)
                                                    app.py, similarity/   (from the repo)
                                                    runs as user 10001 on port 8000; HEALTHCHECK /health
```

| Choice | Why |
|---|---|
| Two stages | The 74 MB raw file and the build steps stay in stage 1; the shipped image holds only what runs |
| Table built inside the image | It always matches the code. `.dockerignore` keeps the laptop's `data/processed/` out, so a stale copy cannot slip in. |
| `requirements.txt` copied before the code | Docker reuses each finished step until its inputs change, so a code edit does not reinstall packages |
| Ordinary user 10001, not root | Many clusters refuse containers that run as root. Kubernetes' `runAsNonRoot` check needs a number, not a name. |
| `HEALTHCHECK` on `/health` | `docker ps` shows the container as healthy or unhealthy. Kubernetes ignores it and uses its own probes. |
| One uvicorn process per container | More traffic means more containers. Extra processes inside one container would each hold their own copy of the data and cache (about 270 MB each), and Kubernetes could not see or restart them one by one. |
| Data inside the image, no volume | It is read-only and 3.3 MB, so any copy can start on any machine with nothing else to set up |

**Publishing the port.** `EXPOSE 8000` only records which port the app uses; it opens nothing. `-p 8000:8000` in `docker run` is what connects the laptop's port 8000 to the container's. Starting the image from Docker Desktop's **Run** button with the optional "Host port" box empty gives a container that is running and healthy but that the browser cannot reach: `docker ps` then shows `8000/tcp` instead of `0.0.0.0:8000->8000/tcp`.

`requirements.txt` now holds only what the image runs: fastapi, uvicorn, numpy, pandas, scikit-learn, pyarrow. `requirements-dev.txt` adds pytest, httpx (for FastAPI's test client) and matplotlib. torch, torchvision and pillow were removed: nothing imports them yet, and they would add over 1 GB to the image. Stage 8 decides where they belong.

### What is inside the image

Measured with `docker history` (one line per Dockerfile step) and `du` inside a running container:

| Part | Size | Comes from |
|---|---:|---|
| Debian Linux, slim | 88 MB | `FROM python:3.10-slim` |
| Python 3.10 + certificates and time zones | 51 MB | `FROM python:3.10-slim` |
| Python packages (`/opt/venv`) | 542 MB | `COPY --from=build /opt/venv` |
|   pyarrow | ~160 MB | Reading the one 3.3 MB Parquet file |
|   scipy (+ its maths libraries) | ~145 MB | Sparse matrices; also required by scikit-learn |
|   numpy (+ its maths libraries) | ~70 MB | Every array |
|   pandas | ~70 MB | Loading the table |
|   scikit-learn | ~50 MB | `TfidfVectorizer` |
|   pip + setuptools | ~25 MB | Come with every venv; not used at runtime |
|   fastapi, uvicorn, pydantic and the rest | ~25 MB | The web layer |
| Our data | 3.3 MB | `products.parquet` |
| Our code | 0.06 MB | `app.py`, `similarity/` |
| **Total** | **684 MB** | |

- Our own code and data are 0.5% of the image. The rest is the Python data stack, which is why images like this are usually 500 MB to 1 GB.
- Across the packages, about 85 MB is their own test suites and about 75 MB is precompiled `.pyc` files. The `.pyc` files are worth keeping, because they make imports at startup faster.
- Docker Desktop lists the image as **890 MB "disk usage"**: it stores both the compressed layers (206 MB) and the unpacked files (684 MB). The **206 MB** is what a registry stores and what each Kubernetes node downloads once.

### How it was checked

| Check | Result |
|---|---|
| `pytest` | 88 passed |
| `docker build`, then `docker run` and calls with curl | Ready 3.6 s after start; 34 ms for a new product, 5 ms for a cached one; 404 and 422 as in the table above; Docker reports the container as healthy; runs as user 10001; 278 MB of memory |
| Same answer inside the image as on this machine | The same top 3 `uniq_id`s, although the image has Python 3.10 and scipy 1.15.3 |
| Before Docker was installed: the image's contents rebuilt in a temporary folder and served by uvicorn | Same status codes; 235 MB of memory on Windows |
| Can every package install on Linux with Python 3.10? (`pip install --dry-run --platform manylinux… --python-version 3.10 --only-binary=:all:`) | All 26 packages, direct and indirect, have prebuilt wheels, so the build needs no compiler |

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| `app.py` at the repo root | Matches the README's sample and its `uvicorn app:app` | One module outside `similarity/`; it holds only the web layer |
| Return a plain list of IDs | The README's signature is `-> List[str]` | No scores or names in the answer; adding them later changes the format |
| `num_similar` at most 100 | Bounds the work and the size of every answer; no page shows 100 recommendations | A caller wanting more gets 422 |
| Check for the ID before searching, instead of catching `KeyError` | A `KeyError` caused by a bug inside the search would otherwise be reported as "not found" | — |
| Plain 500 message | A traceback can reveal code and file paths; it goes to the log instead | The caller learns nothing about the cause |
| Load at startup, not on the first request | The first caller does not wait; `/health` answering means the server can really answer | Startup takes 2.8 s |
| Cache inside the process, not a shared one such as Redis | A miss costs only 22 ms; no extra service to run | One cache per container |
| `def`, not `async def`, for the endpoint | The search is numpy work, not waiting on a network. FastAPI runs a `def` endpoint in a thread pool, so one slow request does not block others. | — |

### Open issues

- **The image could be smaller.** Removing pip, setuptools and the packages' test folders in the build stage would save about 110 MB. Saving the table in a format pandas reads without pyarrow would save about 160 MB more, but Parquet keeps the types and lists (see stage 1). The 206 MB compressed size is downloaded once per machine, so this is not urgent.
- **Python 3.10 reached end of life in October 2026** and gets no more security fixes. It is kept because the README's sample uses it. Moving to 3.12+ is a one-line change in the Dockerfile and would also allow current releases (pyarrow 26+, NumPy 2.3+).
- **Only direct packages are pinned.** For Python 3.10, pip picks scipy 1.15.3 and starlette 1.7.0. This machine has scipy 1.18.1 and starlette 1.10.0, and the tests ran on those. A lock file (for example from `pip-compile`) would pin all 26.
- starlette 1.10 warns that its test client will move from `httpx` to `httpx2`. This affects tests only.

---

## Stage 4b: Kubernetes (optional)

| | |
|---|---|
| Files | `k8s/deployment.yaml`, `k8s/service.yaml` |
| Cluster | Docker Desktop's built-in one-machine cluster (kubeadm), Kubernetes 1.36 |
| Run | `docker tag product-similarity product-similarity:1.0`, then `kubectl apply -f k8s/` |
| Address | `http://localhost:30080/docs` |
| Notebook | None. Kubernetes is driven from the terminal; the commands and their results are below. |

### How it fits together

```
browser ──► localhost:30080 ──► Service "product-similarity" ──┬──► pod 1 ─ container from product-similarity:1.0
                                spreads requests over          └──► pod 2 ─ container from product-similarity:1.0
                                the pods that are ready
                                                   ▲
            Deployment "product-similarity": "keep 2 pods running"; checks /health on each
```

| Term | Meaning |
|---|---|
| Cluster, node | The machines Kubernetes manages; here one, `docker-desktop` |
| Pod | One running copy of the container, with its own address |
| Deployment | The wanted state ("2 pods of this image"). Kubernetes keeps comparing reality with it and fixes any difference. |
| Service | One fixed address in front of the pods |
| Readiness probe | Is the pod ready for traffic? If not, the Service skips it. |
| Liveness probe | Is the pod stuck? If so, it is restarted. |

**Why YAML files here and not for the image?** The Dockerfile says what goes inside the image and runs once, at build time. The YAML says how to run it: how many copies, how much memory, which address, which checks. Kubernetes keeps making that true for as long as it runs. With plain Docker this information was the `docker run -p 8000:8000 …` command line; for Kubernetes there is too much of it for flags. As files in git, it can be reviewed, repeated on any cluster, and applied again safely: `kubectl apply` changes only what differs.

### What the files say

| Setting | Value | Why |
|---|---|---|
| Copies | 2 | One can fail or be replaced while the other answers |
| Image | `product-similarity:1.0`, `imagePullPolicy: IfNotPresent` | Use the image built on this machine. With the tag `latest`, Kubernetes tries to download the image from Docker Hub, where it does not exist. |
| Memory | Reserve 320 MiB, limit 512 MiB | Measured about 270 MiB. Above the limit the pod is restarted, which protects everything else on the machine. |
| CPU | Reserve 0.25 CPU, limit 1 | A search uses one core |
| Readiness | `/health` every 5 s; out after 2 failures | No traffic until the data is loaded |
| Liveness | `/health` every 10 s after 15 s; restart after 3 failures | Restarts a frozen app |
| Rolling update | One extra pod at a time; never fewer ready than wanted | Updates without downtime |
| Before stopping | Wait 5 s (`preStop`) | The Service stops sending requests before uvicorn shuts down |
| Security | Non-root user 10001, read-only file system, no Linux privileges | The app never writes anything, so it can be fully locked down |
| Service | NodePort 30080 → container port 8000 | Docker Desktop opens NodePorts on `localhost`; requests are spread over all ready pods |

### Experiments

Each run sent requests to `localhost:30080` back to back, one at a time, each on a new connection, while the pods were disrupted (`k8s_load.py`, a scratch script).

| Experiment | How | Result |
|---|---|---|
| Deploy | `kubectl apply -f k8s/` | Both pods ready after 7.5 s. The first readiness probe was refused while the data loaded, so no request reached a pod too early. |
| Same answers | `curl` through the Service | The same top 3 as Docker and this machine; 404 and 422 as before; 37 ms for a new product |
| Load balancing | 10 s of requests | 627 and 620 of 1,247 requests per pod; none failed |
| Pod deleted | `kubectl delete pod` | A replacement was ready within 7.5 s; **0 of 1,731 requests failed** |
| App crashed | `kubectl exec <pod> -- sh -c "kill 1"` | Restarted by itself (RESTARTS 1). **47 of 1,570 requests failed (3%)**, all within 0.9 s. |
| App frozen | `docker pause` on the pod's container | Readiness took it out of the Service after about 10 s; liveness restarted it after about 30 s. 2 requests timed out. The test client sends one request at a time and waited 5 s on each; with many users at once, about half the requests in those 10 s would have hung. |
| Update | `kubectl rollout restart` | Both pods replaced in about 14 s; **0 of 2,146 requests failed** |
| Scaling | `kubectl scale --replicas=3` | The third pod was ready in 6.7 s; 298 / 305 / 331 requests per pod. `kubectl apply -f k8s/` then set it back to 2. |
| Memory | `docker stats` | About 270 MiB per pod, of the 512 MiB limit |

**Planned vs unplanned stops.** When Kubernetes stops a pod itself (a delete, an update, scaling down), it first takes the pod out of the Service, waits 5 s, and only then stops the app. Nothing was lost. A crash gives no warning: for about a second the Service still sends requests to a pod whose app is gone. A frozen app is worse, because it receives requests until readiness fails twice, up to about 10 s.

### Operating it

Deploying takes three commands. Everything in the experiments above was a test, not a step.

```
docker build -t product-similarity .                          # make the image
docker tag product-similarity:latest product-similarity:1.0   # give it a version
kubectl apply -f k8s/                                         # hand the two YAML files to Kubernetes
```

`kubectl` is the command-line remote control for Kubernetes, as `docker` is for Docker.

| To | Do |
|---|---|
| Use it | Open `http://localhost:30080/docs`, about 10 s after starting |
| See the pods | `kubectl get pods`: both should be `1/1 Running` |
| See one pod's requests | `kubectl logs <pod name>` |
| Remove the app | `kubectl delete -f k8s/` (frees about 540 MiB) |
| Ship a code change | `docker build -t product-similarity:1.1 .`, change the tag in `k8s/deployment.yaml`, `kubectl apply -f k8s/` |
| Pause all of Kubernetes | Docker Desktop's **Stop cluster**. **Start cluster** brings our 2 pods back by themselves, because Kubernetes keeps the applied instructions in its own database. |
| Wipe Kubernetes | Docker Desktop's **Reset cluster**. Our app is gone until `kubectl apply -f k8s/` is run again. |

**Why the pods appear in Docker Desktop.** Kubernetes does not run containers itself; it tells a container engine what to run. On this laptop that engine is Docker, so each pod's container is an ordinary Docker container, named `k8s_api_<pod name>_…` (`k8s_` = started by Kubernetes, `api` = the container's name in the YAML). The 2 containers are the 2 copies from `replicas: 2`. Deleting one in Docker Desktop does not remove it: Kubernetes sees one copy missing and starts another. To really stop them, use `kubectl delete -f k8s/` or Stop cluster.

**More than one Kubernetes.** One cluster normally runs many apps, each with its own YAML files, optionally grouped into namespaces (for example `dev` and `test`). Several clusters on one laptop are possible with tools such as kind or minikube; `kubectl config use-context <name>` switches between them. Companies typically keep separate clusters for development, testing and production.

*k8s* is shorthand for Kubernetes: a K, the 8 letters "ubernete", and an s. The name is Greek for the person who steers a ship.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Docker Desktop's cluster (kubeadm) | Already installed; uses the locally built image, so no image registry is needed | One machine: both copies fail together if it does |
| NodePort | Reachable from the browser on Docker Desktop, and spreads requests over pods. `kubectl port-forward` connects to a single pod only. | A cloud cluster would use a LoadBalancer Service or an Ingress instead |
| Version tag instead of `latest` | Kubernetes would otherwise try to download the image | Each new build needs a new tag (1.1, …) in the YAML, or running pods keep the old image |
| The same `/health` for readiness and liveness | The app's only states are "loading" and "answering"; it depends on nothing outside itself | A liveness check that also tested an outside service would restart healthy pods when that service fails, so it is kept self-contained |
| 5 s wait before stopping | Covers the delay before the Service stops routing to a stopping pod | Every pod takes 5 s longer to stop. The test was not repeated without it. |
| No autoscaler, no Ingress | Not needed for a local demonstration; autoscaling also needs metrics-server installed | Scaling is manual |

### Open issues

- A crash loses about one second of requests before the Service notices. Kubernetes cannot prevent this alone; callers should retry once on a connection error.
- Probes run every 5 to 10 s, so a frozen pod can receive requests for up to about 10 s.
- Both copies run on one machine. On a multi-node cluster, a `topologySpreadConstraints` rule would put them on different nodes.
- The YAML pins `product-similarity:1.0`; a new version means a new tag and an edit to `k8s/deployment.yaml`.

---

## Stage 5: Evaluation

| | |
|---|---|
| Code | `evals/shortlist.py` picks the 20 test products and gathers a first shortlist to label. `evals/sweep.py` then searches the whole catalogue for each test product's kind of product. Both were run once. `evals/run_eval.py` scores the methods and rebuilds the workbook. `evals/store.py` reads and writes the data sheets. |
| Data | Everything is in `evals/eval_results.xlsx`, with no separate files. The **Queries** sheet holds the 20 test products. The **Labels** sheet holds the 3,841 hand labels, with the ground truth and where each candidate came from (`source`: shortlist or sweep). |
| Run | `python -m evals.run_eval` rebuilds `evals/eval_results.xlsx` and prints the same numbers (about 10 s). Close the workbook in Excel first. |
| Tests | `tests/test_evals.py` (7 tests; 95 across all stages) |
| Notebook | None. The run produces the Excel workbook. |

### Method

1. **Pick 20 test products** (queries). 12 are random (fixed seed): 10 spread over the 8 largest categories, plus 2 from any other category. 8 are hard cases chosen for a known difficulty (table below).
2. **Round one: a shortlist per test product.** 28 to 46 candidates, 736 in total, from four routes. They were shuffled so the reviewer could not tell which route found which:

   | Route | Candidates |
   |---|---|
   | Current method | top 15 |
   | Title words (TF-IDF, stage 2) | top 15 |
   | Parts of words (character TF-IDF, which links kurta/kurti, tee/t-shirt, frok/frock) | top 10 |
   | Same group: random products of the same category and gender at 0.5x to 2x the price | 10 |

3. **Round two: a full-catalogue sweep per test product.**
   - **Terms:** words from its own title that describe its kind of product (garment type, audience, a key detail), plus a price band around its own price. They never come from any method's results.
   - **Search:** the terms are matched against all 30,000 titles, and every match not already labelled is read: 57 to 209 per test product, 3,105 in total.
   - **Size:** the terms were tightened until each sweep was about 150 to 200 products. Where fewer matched, all were read.
   - **Record:** the exact terms are in `evals/sweep.py`.
4. **Label by hand.** Every candidate from both rounds is marked good or not good, with one rule (below).
5. **Pick the ground truth:** the 10 most similar good candidates from both rounds, best first.
6. **Score** the method's top 10 against the labels.

**Why the shortlist routes, and then a sweep:**
- **Why not just read the current method's results:** nobody can read 30,000 products for each test product, so the answer key has to come from a smaller set. If that set were the current method's own results, the "right answers" would be its own answers, like an answer key copied from the student's paper. It could never be caught missing a better product.
- **What the shortlist added:** the four routes put products the method misses in front of the reviewer. But the shortlist still only showed what those routes ranked highly. For the blouse fabric, every route leaned on its wrong category (kurtis), so not one fabric was read.
- **What the sweep adds:** it reads every product of the right kind. The ground truth becomes the best of everything of that kind, not the best of a shortlist.

| Hard case | Why it is hard |
|---|---|
| Q13 Kalamkari saree blouse fabric | Fabric filed under kurtis |
| Q14 Baby girls' cotton printed frock | Similar items are often called "dress" |
| Q15 Women's straight kurta | Similar items are often called "kurti" |
| Q16 Marvel Hydra cuff beanie | Only 2 attributes (no category, price or gender) |
| Q17 Toddler "Straight Outta Timeout" T-shirt | No category, no price |
| Q18 "Pari Singh Orange Plain Women's **Frok**" | Misspelled, no price |
| Q19 "OOMPH!" (a saree) | The title is only the brand |
| Q20 Captain America boys' T-shirt | The same product listed under 3 `uniq_id`s |

| Test product | The sweep searched for | Read | New good | Ground-truth picks from the sweep |
|---|---|---:|---:|---:|
| Q01 Idalia kurta set with dupatta | Women's kurtas with set / dupatta / palazzo / salwar / sharara, 0.5x–2x price | 127 | 120 | 6 |
| Q02 Block-print anarkali kurti | Women's kurtis, block print or cotton / printed anarkali, 0.5x–2x | 134 | 134 | 4 |
| Q03 Men's plain regular T-shirt | Men's plain / solid regular-fit T-shirts, 0.75x–1.33x | 181 | 181 | 4 |
| Q04 Men's plain slim T-shirt | Men's plain / solid slim-fit T-shirts, 0.5x–2x | 163 | 160 | 4 |
| Q05 Silk satin saree, ₹3,669 | Silk or satin sarees, 0.4x–2.5x | 201 | 199 | 6 |
| Q06 Linen formal shirt | Men's linen shirts, or plain regular-fit formal shirts, 0.5x–2x | 194 | 187 | 3 |
| Q07 Girls' cotton frock | Girls' cotton frocks and dresses, 0.5x–2x | 193 | 177 | 5 |
| Q08 Printed casual shirt | Men's printed casual shirts, 1/3x–3x | 134 | 134 | 5 |
| Q09 Max boys' T-shirt | Boys' regular-fit T-shirts, 0.85x–1.18x | 175 | 153 | 0 |
| Q10 Synthetic a-line dress | Women's a-line dresses, 0.67x–1.5x | 197 | 186 | 3 |
| Q11 Wool newsboy cap | Newsboy / flat / ivy / cabbie / gatsby / beret caps, not women's | 57 | 35 | 7 |
| Q12 Jockey poco brief | Men's cotton briefs (not boxer briefs or trunks), packs or poco | 196 | 157 | 7 |
| Q13 Kalamkari blouse fabric | Women's fabrics and the fabric category, not sarees or suit sets, 1/3x–3x | 64 | 10 | 8 |
| Q14 Baby girls' frock | Frocks and dresses for babies or toddlers, not boys', 0.5x–2x | 176 | 104 | 4 |
| Q15 Straight kurta, ₹379 | Women's straight kurtas, cotton / rayon / Rajasthani, 0.75x–1.33x | 159 | 155 | 2 |
| Q16 Marvel cuff beanie | Beanies and knit caps that are cuffed, knit or have a logo | 156 | 104 | 6 |
| Q17 Toddler slogan T-shirt | Toddler and baby T-shirts, not girls' | 100 | 50 | 6 |
| Q18 "Frok" | Women's plain / solid / orange / shift dresses, not kurtas | 101 | 73 | 6 |
| Q19 "OOMPH!" | Every OOMPH! saree, plus silk sarees at 0.92x–1.08x | 209 | 207 | 1 |
| Q20 Captain America boys' T-shirt | Boys' character T-shirts, or plain regular-fit ones near its price | 188 | 134 | 4 |
| **Total** | | **3,105** | **2,660** | **91** |

Word matching pulls in some noise, and all of it was labelled not good:
- **"baby":** brought 50+ women's babydoll lingerie items into Q14.
- **The T-shirt pattern** also matches "fi**t Shirt**", so some collared shirts were read too.

### What counts as good

A shopper looking at the test product would accept it as a similar alternative:
- the same kind of product, with synonyms counting as the same (kurta = kurti, frock = dress, tee = T-shirt);
- for the same audience (men, women, boys or girls);
- a broadly similar style;
- a price within about 3x when both prices are known.

Multipacks of 5 or more count as a different purchase. The sweep needed some calls made explicit, each recorded in the labelling notes:
- Berets are not newsboy caps.
- Nighties, maternity dresses and unstitched dress materials are not dresses.
- Sweatshirts, onesies and tee-and-pants sets are not T-shirts.
- Kids' beanies are not an adult beanie.
- Half sarees are not sarees.
- Boxer briefs and gym supporters are not briefs.

Labels were made from titles and attributes, not photos.

3,841 candidates are labelled: 736 from round one (their labels unchanged) and 3,105 from the sweep. 3,179 (83%) are good. Every test product now has a full ground truth of 10. After round one, Q11 had only 8 and Q13 only 2.

### How the ground truth changed

The ground truth now has 200 picks, up from 190: 109 were kept, 81 were replaced by closer products the sweep found, and 91 were added from the sweep. Per test product, the last column of the sweep table above shows how many picks came from the sweep. Examples:
- **Q13, blouse fabric:** from 2 picks to 10, adding hand-block-printed and ikat cotton fabrics.
- **Q12, Jockey Poco brief (style 8035):** seven packs of the same 8035 style in other colours and sizes replaced looser picks.
- **Q11, wool newsboy cap:** seven wool and tweed newsboy and flat caps.
- **Q17, toddler slogan T-shirt:** six toddler slogan T-shirts replaced generic boys' T-shirts.
- **Q18, "Frok":** a rust-orange midi dress and plain cotton dresses.
- **Q09, Max boys' T-shirt:** unchanged. Dozens of identical Max listings, and the picks were already the closest.

### The three metrics

| Metric | Meaning | Best |
|---|---|---|
| **Good results** | Of the method's 10 results, how many are labelled good | 10 of 10 |
| **Ground truth found** | Of the 10 hand-picked closest products, how many the method also returned | 10 of 10 |
| **First good result** | The rank of the first good result ("none" counts as 11 in the average) | 1 |

A result nobody has labelled is shown as **not labelled**, and the run prints a warning.

The first two metrics answer different questions. Good results asks "are the results acceptable?". Ground truth found asks "are they the closest available?". Near-copies keep the second one low even for a good method: Q09's catalogue has dozens of listings titled "Max Boy's Regular Fit T-Shirt".

### Results

Current method (stage 3: TF-IDF title + other attributes):

| | Round-one ground truth | Final ground truth |
|---|---:|---:|
| Good results (of 10) | 9.0 | **9.0** |
| ... random test products | 9.3 | **9.3** |
| ... hard test products | 8.6 | **8.6** |
| Ground-truth picks found | 52% (99 of 190) | **32% (64 of 200)** |
| Average first good result | 1.0 | **1.0** |
| Test products whose first result is good | 20 of 20 | **20 of 20** |

| Test product | Good (of 10) | Ground truth found: round one | Ground truth found: final |
|---|---:|---:|---:|
| Q01 Idalia printed kurta set | 10 | 5 | 4 |
| Q02 Zoeyams block print anarkali kurti | 10 | 6 | 4 |
| Q03 Indian Terrain men's plain T-shirt | 10 | 7 | 4 |
| Q04 Flying Machine men's slim T-shirt | 10 | 7 | 5 |
| Q05 Banarasi silk satin saree, ₹3,669 | 10 | 3 | 2 |
| Q06 BLUEPOCKET linen formal shirt | 10 | 1 | 1 |
| Q07 Arshia girls' cotton frock | 10 | 5 | 3 |
| Q08 US Polo printed casual shirt | 10 | 4 | 1 |
| Q09 Max boys' T-shirt | 10 | 5 | 5 |
| Q10 RARE synthetic a-line dress | 10 | 6 | 5 |
| Q11 SIGGI wool newsboy cap | **1** | 1 of 8 | **0** |
| Q12 Jockey poco brief | 10 | 7 | 3 |
| Q13 Kalamkari blouse fabric | **1** | 1 of 2 | 1 |
| Q14 Zadmus baby girls' frock | 8 | 4 | 3 |
| Q15 Rangeelo straight kurta | 10 | 9 | 8 |
| Q16 Marvel cuff beanie | 10 | 7 | 3 |
| Q17 Toddler slogan T-shirt | 10 | 4 | **0** |
| Q18 "Frok" | 10 | 6 | 3 |
| Q19 "OOMPH!" | 10 | 7 | 6 |
| Q20 Captain America boys' T-shirt | 10 | 4 | 3 |

**Findings**

- **The results are acceptable, and that did not change.**
  - **Numbers:** good results stay at 9.0, all 10 results are good for 17 of the 20 test products, and the first result is good for all 20.
  - **Why they stayed:** every product in the method's top 10 was already labelled in round one.
- **The results are often not the closest available.** Ground truth found fell from 52% to 32%, because the sweep found closer products the method does not return. Typical reasons:
  - **Brand outvotes the details.** Q08: 9 of the 10 results are US Polo shirts, mostly solid, so the printed regular-fit casual shirts of other brands are missed (1 found).
  - **Close variants are missed.** Q12: other Jockey brief styles are shown instead of the same 8035 Poco style in other colours and sizes (3 found).
  - **The finer meaning of the title is lost.**
    - Q17: generic boys' T-shirts are shown, and none of the toddler slogan T-shirts (0 found).
    - Q16: plain cuff beanies are shown instead of other licensed-logo beanies such as Star Wars, Fallout and Subaru (3 found).
- **Two clear failures remain, both caused by attributes outvoting the title:**
  - **Q13, kalamkari fabric:** 1 good result. The other 9 come from its wrong category: 8 kurtis and a cape.
  - **Q11, wool newsboy cap:** 1 good result and 0 of the 10 closest. The results are the same brand's women's caps and sporty caps from the same category.
- **Text-only comparison:** on the round-one ground truth, a text-only baseline (TF-IDF on titles) scored 6.3 good results against 9.0, and 3.0 against 8.6 on the hard test products. Combining attributes is what rescues titles that mislead: the Marvel beanie, "Frok" and "OOMPH!".

### Ranking metrics: one score row per method

The **Methods** sheet gives every method in `evals/run_eval.py` one row with standard ranking metrics, so methods can be compared side by side. Each test product's 10 hand picks are the right answers, and every metric is averaged over the 20 test products:

| Metric | Meaning | Current method |
|---|---|---:|
| **Recall@5** | Share of the 10 picks in the top 5 (at most 50%, since 5 slots cannot hold 10) | 24% |
| **Recall@10** | Share of the 10 picks in the top 10 | 32% |
| **MRR@10** | 1 ÷ rank of the first pick, averaged; 1.0 means a pick always comes first | 0.82 |
| **Good@10** | Share of the top 10 labelled good: the only metric using the good / not good labels | 90% |
| **ms per query** | Search time per test product (latency) | 22 |

Precision@5, hit rate and nDCG were tried and dropped to keep the table short. With exactly 10 picks per test product they repeat what recall and MRR already show.

**In short:** the first result is usually one of the closest products (MRR 0.82), but only about a third of the closest 10 make it into the top 10 (Recall@10 32%).

The metrics against the hand picks need no new labelling, so any new method gets them immediately. Only Good@10 needs its results labelled first. The numbers are computed by the script (`ranking_metrics` in `evals/run_eval.py`, tested against a hand-worked example) and written as values. Hovering over a column header shows its meaning.

### How to read the workbook

- **Methods:** one row per method with the ranking metrics above.
- **Summary:** one row per test product with good results, ground truth found and first good result for the current method, then the overall numbers. These are Excel formulas over Details.
- **Details:** 10 rows per test product. The ground truth sits beside the method's top 10, and each result is marked Good? and In ground truth? (green / red).
- **Queries, Labels:** the data. The yellow cells (`good`, `gt_rank`) are the ones to edit. A change made there is kept the next time `python -m evals.run_eval` runs.
- **Good** means a shopper would accept it as a similar alternative: same kind of product, same audience, similar style, price within about 3x.

Excel recalculated all 86 formulas with no errors, and every value matched the script's own Python calculation.

**Adding a method** (for example sentence embeddings in stage 6):
1. Add it to `METHODS` in `evals/run_eval.py`.
2. Run `python -m evals.run_eval`. It gets a row on the Methods sheet. Add `--alternative <name>` to also show its results beside the current method in Summary and Details.
3. Its results that were never labelled are added to the Labels sheet with an empty `good` cell, and the run warns about them. Fill in yes / no and run again; until then only Good@10 is incomplete.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Hand-labelled once, in this session, instead of an LLM API judge | No API key or cost, and simpler. The labels are saved, so every method is scored on identical ground. | One annotator, and an AI. Worth spot-checking a few test products. |
| Shortlist from four routes, then a full-catalogue sweep | The sweep removes the dependence on what the routes rank highly. Its terms come from each product's own title, not from any method. | The terms are judgement. A matching product that uses none of the words is still missed. |
| Sweep tightened to about 150–200 products per test product | For common kinds (kurtas, sarees) hundreds of products are near-equal, so reading 1,400 would mostly reorder equals | The ground truth is the best of the closest 150–200, not of every product of that kind |
| Good/not good for every candidate, plus a ranked top 10 | A strict ID match alone would punish equally good near-copies | Two numbers to read instead of one |
| Price only as a 3x limit | Keeps "good" about what the product is | Price closeness is barely tested |
| Unlabelled results shown as "not labelled", with a warning | A new method's unseen results are neither right nor wrong until labelled | They must be labelled before its numbers are final |
| All data in the workbook, no separate files | One file to open, share and edit. The Labels sheet is read back on every run, so edits made in Excel are kept. | The labels live in a binary file: git cannot show which label changed. Excel must be closed while the script runs. |
| Ranking metrics against the hand picks, in their own sheet | Standard numbers (recall, MRR) that score any method straight away, with no new labelling; plus Good@10 and latency | Near-copies keep recall low even for a good method, so Good@10 is reported next to it |
| Methods metrics as values; Summary as formulas | The Methods sheet scores every registered method, including ones not shown in Details, so it cannot be formulas over Details | Editing a label in Excel updates Summary immediately, but Methods only after a rerun |
| Own folder `evals/`, no notebook | Evaluation stays apart from the service; one command produces the workbook | — |

### Open issues

- **One annotator,** labelling from text, not photos. The uncertain calls are listed above, under what counts as good, and can be corrected on the Labels sheet.
- **Good results is near its ceiling** for the current method. Stage 6 improvements will show mainly in ground truth found and on the hard test products. With 20 test products, only large differences are meaningful.
- **The sweep's T-shirt pattern also matched "fit Shirt".** Some collared shirts were read (and labelled not good), and a few "...fit Shirt" titles were left out of the Q06 and Q08 sweeps. `evals/sweep.py` keeps the pattern as run, with a note.
- **Every result from the current method is labelled,** because its top 15 was one of the shortlist routes. Another method's results may need labelling first.

## Upcoming stages

Four methods, each a row on the Methods sheet. They share the stage 3 search and differ only in their weights:

| Attribute | `tfidf` (current) | `minilm` | `tfidf_image` | `minilm_image` |
|---|---|---|---|---|
| Title, TF-IDF | 0.30 | — | 0.30 | — |
| Title, all-MiniLM-L6-v2 | — | 0.30 | — | 0.30 |
| Category, gender, price, brand, colour, weight | 0.20, 0.10, 0.15, 0.15, 0.05, 0.05 | same | same | same |
| Photo, CLIP ViT-B/32 | — | — | 0.20 | 0.20 |

- **6 Title embeddings**: all-MiniLM-L6-v2 (Reimers & Gurevych, 2019) reads the same product name as TF-IDF and gives each title 384 numbers, so titles with the same meaning but different words (frock / dress) score high. Embedding cosines are rescaled so a typical random pair scores 0: `max(0, (cosine − b) / (1 − b))`, with `b` the median of 10,000 random pairs.
- **7 Images (optional)**: the main `medium` photo of each product, embedded with CLIP ViT-B/32 (Radford et al., 2021), with the same rescaling. A pair where either product has no photo skips the photo score (the fallback). CLIP is a vision transformer, not the CNN the brief names; it was chosen because it learned from images with captions, while ImageNet CNNs learned 1,000 object classes with almost no clothing.
  - **The photos are stored outside the repo**, in `C:\Users\aswin\Downloads\Amazon Image Download`. The repo is inside OneDrive, and 30,000 photos would otherwise sync to the cloud. Only the photo numbers (`image_embeddings.npz`) are kept in `data/processed/`.
  - **The download is standalone.** Only `python -m similarity.images download` fetches photos. Preprocessing, the Docker build, the API and the evaluation never do.
- **8 ANN index (bonus)**: HNSW (Malkov & Yashunin, 2016), measured by recall@k against stage 3.
