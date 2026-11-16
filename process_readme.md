# Process: product similarity search

A running record of what each stage found, what it does, and why. It is updated as each stage is finished. The exercise itself is in [Problem_Statement.md](Problem_Statement.md); the short overview is [README.md](README.md).

Data: `data/archive/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`

Stages 1 to 3 and 6 to 8 each have a module in `similarity/` that holds the logic and a numbered notebook in `notebooks/` that walks through it. A notebook's first cell moves to the repo root, so it runs the same whether opened from `notebooks/` or the root. Notebooks call the module's functions; they never copy its logic. Stage 4 adds the web layer (`app.py`, `Dockerfile`) and stage 4b the Kubernetes files (`k8s/`). The evaluation (stage 5) is kept apart in `evals/`. Run everything from the repo root.

## Pipeline

```
raw ldjson ─► 01 preprocessing ─► data/processed/products.parquet ─► 02 features ─► 03 similarity search ─► 04 API + Docker ─► 4b Kubernetes
                                                                                 ├─► 05 evaluation (scores every method)
                                                                                 ├─► 06 title embeddings (MiniLM)
                                                                                 ├─► 07 images (optional, CLIP)
                                                                                 └─► 08 ANN index
```

| Stage | Module | Notebook | Output | Status |
|---|---|---|---|---|
| 0 EDA | — | `notebooks/eda.ipynb` | Findings below | Done |
| 1 Preprocessing | `similarity/preprocessing.py` | `notebooks/01_preprocessing.ipynb` | `data/processed/products.parquet` | Done |
| 2 Features | `similarity/features.py` | `notebooks/02_features.ipynb` | `Features` object, built in memory | Done |
| 3 Similarity search | `similarity/search.py` | `notebooks/03_search.ipynb` | `find_similar_products` | Done |
| 4 API + Docker | `app.py`, `Dockerfile` | `notebooks/04_api.ipynb` | `GET /find_similar_products`, Docker image | Done |
| 4b Kubernetes (optional) | `k8s/` | — (terminal commands below) | Deployment + Service on Docker Desktop | Done |
| 5 Evaluation | `evals/` | — (`python -m evals.run_eval` writes the workbook) | `evals/eval_results.xlsx` | Done |
| 6 Title embeddings | `similarity/embeddings.py` | `notebooks/06_text_embeddings.ipynb` | `data/processed/title_embeddings.npz`, method `minilm` | Done |
| 7 Images (optional) | `similarity/images.py` | `notebooks/07_image_similarity.ipynb` | Photos (outside the repo), `data/processed/image_embeddings.npz`, methods `tfidf_image`, `minilm_image` | Done |
| 8 ANN index | `similarity/ann.py` | `notebooks/08_ann.ipynb` | `data/processed/product_hnsw.faiss`, method `tfidf_ann` | Done |

When each part runs:

| When | What | Time |
|---|---|---|
| Once, when the Docker image is built | Stage 1 parses the raw file into `products.parquet` | about 3 s |
| Once, when the server starts | Stage 2 turns the table into arrays: TF-IDF matrix, codes, numbers | about 0.5 s (2.8 s with imports) |
| A request for a new product | Stage 3 scores it against all 30,000 products; the best 100 are kept | about 22 ms in Python, 38 ms over HTTP |
| A request for a product asked before | The kept 100 are cut to `num_similar` | about 2 ms over HTTP |
| Once, by hand, offline (stage 6) | Encode the 30,000 titles with MiniLM into `title_embeddings.npz` | 226 s on 4 CPU threads |
| Once, by hand, offline (stage 7) | Download the 30,000 main photos, outside the repo | about 13 min |
| Once, by hand, offline (stage 7) | Encode the 29,861 photos with CLIP into `image_embeddings.npz` | about 27 min of encoding |
| Once, by hand, offline (stage 8) | Build the HNSW index of the 30,000 product vectors into `product_hnsw.faiss` | 63 s on 1 thread |
| By hand, to measure (stage 8) | The scale test: 1,000,000 made-up vectors, exact search vs HNSW | about 13 min, 2 GB of memory |

## Settings

Everything that can change without editing code is a setting: which method the API uses, the weights, the ANN settings, file paths, models and download limits. All of them, with their defaults and what each does, are in [`.env.example`](.env.example); `similarity/config.py` reads them.

| Where a setting comes from | Used when |
|---|---|
| 1. An environment variable of the same name | Always wins: `docker run -e`, the Kubernetes ConfigMap, or the terminal |
| 2. `.env` in the repo root | Local runs. Copy `.env.example` to `.env` and edit it; `.env` is kept out of git. |
| 3. The default in `similarity/config.py` | When neither sets it (the same values as `.env.example`) |

Settings are read when Python starts, so restart the API, or rerun the command, after a change. A bad value stops the start with its name, for example `setting SIMILARITY_METHOD='word2vec': choose one of tfidf, minilm, tfidf_image, minilm_image`. The tests ignore `.env` and always run with the defaults.

### All settings and their options

**What "after a change" means:**
- **Restart:** restart the API, or rerun the command that uses the setting.
- **Rebuild the index:** run `python -m similarity.ann build`. Needed only if you use `SEARCH_MODE=ann`; a stale index refuses to load.
- **Paths** are relative to the folder commands are run from, which is the repo root. `~` is your home folder.

**The API**

| Setting | Default | Options | What it changes | After a change |
|---|---|---|---|---|
| `SIMILARITY_METHOD` | `tfidf` | `tfidf`, `minilm`, `tfidf_image`, `minilm_image` | Which score the API ranks by (stages 3, 6, 7) | Restart |
| `SEARCH_MODE` | `exact` | `exact`, `ann` | Score every product, or an HNSW shortlist then the same score (stage 8) | Restart |
| `MAX_SIMILAR` | `100` | A whole number, 1 or more | The largest `num_similar` a caller may ask for, and how many results are cached per product | Restart |
| `CACHE_SIZE` | `1024` | A whole number, 0 or more (0 turns the cache off) | How many products' results the API keeps in memory | Restart |

**The score**

Weights are any number of 0 or more. They need not add up to 1, since a pair is judged only on the attributes both products have, and 0 leaves that attribute out.

| Setting | Default | What it weighs | After a change |
|---|---|---|---|
| `WEIGHT_TEXT` | `0.30` | The title (TF-IDF, or MiniLM in the `minilm` methods) | Restart; rebuild the index |
| `WEIGHT_CATEGORY` | `0.20` | Same child category (browsenode when there is none) | Restart; rebuild the index |
| `WEIGHT_GENDER` | `0.10` | Same gender | Restart; rebuild the index |
| `WEIGHT_PRICE` | `0.15` | Smaller price ÷ larger price | Restart; rebuild the index |
| `WEIGHT_BRAND` | `0.15` | Same brand | Restart; rebuild the index |
| `WEIGHT_COLOUR` | `0.05` | Colours in common ÷ colours of either | Restart |
| `WEIGHT_WEIGHT` | `0.05` | Smaller weight (grams) ÷ larger | Restart |
| `WEIGHT_IMAGE` | `0.20` | The photo, added on top; used only by `tfidf_image` and `minilm_image` | Restart |

| Setting | Default | Options | What it changes | After a change |
|---|---|---|---|---|
| `SCORE_DECIMALS` | `3` | A whole number, 0 or more | Scores equal to this many decimals count as a tie, and the higher rating goes first | Restart |

**ANN search** (only used when `SEARCH_MODE=ann`)

| Setting | Default | Options | What it changes | After a change |
|---|---|---|---|---|
| `ANN_CANDIDATES` | `1000` | A whole number, 10 or more. Measured recall@10 against exact search: 100 → 88%, 250 → 93%, 500 → 96%, 1,000 → 97%, 2,000 → 98% | How many products HNSW shortlists for the full score. More is closer to exact search, but slower. | Restart |
| `ANN_M` | `64` | A whole number, 2 or more. Tried: 32 (41 products unreachable), 64, 128 | Links per product in the graph. More means fewer unreachable products, but a bigger index and a slower build. | Rebuild the index |
| `ANN_EF_CONSTRUCTION` | `500` | A whole number, 1 or more. Tried: 200, 500 | How many candidates are weighed when a product is linked in. Larger gives a better graph but a slower build. | Rebuild the index |

**Files**

| Setting | Default | Options | What it is | After a change |
|---|---|---|---|---|
| `RAW_DATA_PATH` | `data/archive/marketing_sample_for_amazon_com-…_30k_data.ldjson` | Any file path | The raw file stage 1 reads | `python -m similarity.preprocessing` |
| `PRODUCTS_PATH` | `data/processed/products.parquet` | Any file path | The clean table: written by stage 1, read by everything else | `python -m similarity.preprocessing` |
| `TITLE_EMBEDDINGS_PATH` | `data/processed/title_embeddings.npz` | Any file path | The title numbers of stage 6 | `python -m similarity.embeddings` |
| `IMAGE_EMBEDDINGS_PATH` | `data/processed/image_embeddings.npz` | Any file path | The photo numbers of stage 7 | `python -m similarity.images embed` |
| `ANN_INDEX_PATH` | `data/processed/product_hnsw.faiss` | Any file path | The HNSW index of stage 8 | Rebuild the index |
| `IMAGE_DIR` | `~/Downloads/Amazon Image Download` | Any folder | Where stage 7 keeps the photos, outside the repo | `python -m similarity.images download` |

**The offline steps** (stages 6 and 7)

| Setting | Default | Options | What it changes | After a change |
|---|---|---|---|---|
| `TITLE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model on Hugging Face, for example `sentence-transformers/all-mpnet-base-v2` (768 numbers, about 5 times slower) | The model that turns titles into numbers | `python -m similarity.embeddings`, then rebuild the index |
| `IMAGE_MODEL` | `sentence-transformers/clip-ViT-B-32` | A CLIP model in sentence-transformers: `clip-ViT-B-32`, `clip-ViT-B-16`, `clip-ViT-L-14` (each larger and slower) | The model that turns photos into numbers | `python -m similarity.images embed` |
| `ENCODE_THREADS` | `4` | A whole number from 1 to the CPU's thread count (16 on this laptop) | CPU threads for encoding titles | The next encoding |
| `DOWNLOAD_WORKERS` | `8` | A whole number, 1 or more; keep it low to be polite to the server | Photos downloaded at the same time | The next download |
| `DOWNLOAD_TIMEOUT` | `10` | Seconds, 1 or more | How long to wait for each photo | The next download |

**Switching the API's method** takes two settings. They match the two separate questions of stages 3 to 8: how a product is scored, and how the products to score are found.

| Setting | Values | Default |
|---|---|---|
| `SIMILARITY_METHOD` | `tfidf`, `minilm`, `tfidf_image`, `minilm_image`: which score | `tfidf` |
| `SEARCH_MODE` | `exact` (score every product) or `ann` (an HNSW shortlist, then the same score on it) | `exact` |

| Choice | Also needs |
|---|---|
| `tfidf` | Nothing: the Docker image has it all |
| `minilm` | `data/processed/title_embeddings.npz` (`python -m similarity.embeddings`) |
| `tfidf_image`, `minilm_image` | `data/processed/image_embeddings.npz` (`python -m similarity.images embed`), plus the title file for `minilm_image` |
| `SEARCH_MODE=ann` | faiss-cpu, the title file and `data/processed/product_hnsw.faiss` (`python -m similarity.ann build`) |

At startup the API runs one search, so a missing file stops it straight away, with the command that makes the file, instead of failing on every request. `/health` reports the method in use: `{"status": "ok", "products": 30000, "method": "tfidf", "search": "exact"}`.

- **Locally:** edit `.env` and restart `uvicorn app:app`.
- **Docker:** `docker run --env-file .env -p 8000:8000 product-similarity`. The image holds only what `tfidf` with exact search needs.
- **Kubernetes:** edit `k8s/configmap.yaml`, then `kubectl apply -f k8s/configmap.yaml` and `kubectl rollout restart deployment/product-similarity`. If the image lacks what the new method needs, the new pods stop at startup and never become ready, and the old pods keep serving.

**Weights and the ANN index.** The ANN index is built from the weights. After changing a weight, rebuild it (`python -m similarity.ann build`); a stale index refuses to load.

**What stays in code:** the rules that describe the data, such as colour spellings, the weight sentinel and the brand placeholders. They are facts found in the data, not choices to tune.

---

## Stage 0: EDA

`notebooks/eda.ipynb` reproduces every number in this section. The only charts, two price histograms, are drawn in `notebooks/eda.ipynb`.

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

`sales_price` loads as a number for 27,110 rows (90.4%). Nothing is zero or negative. `notebooks/eda.ipynb` draws a histogram of the price and one of its log10, which shows the skew.

| Min | 5th pct | 25th pct | Median | 75th pct | 95th pct | 99th pct | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| ₹39 | ₹219 | ₹379 | ₹590 | ₹899 | ₹2,877 | ₹5,220 | ₹9,988 |

There is no separate `price` column, even though the problem statement lists both "price" and "sales_price".

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
- 8,087 more rows have no `colour` but a colour word in `product_name` (35-word list in `notebooks/eda.ipynb`). A vocabulary built from the `colour` column does not work: it picks up junk such as `as`, `april` and `b`.

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
| Notebook | `notebooks/01_preprocessing.ipynb` |
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
| Notebook | `notebooks/02_features.ipynb` |
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
| Notebook | `notebooks/03_search.ipynb` |
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

A missing attribute is left out for that pair and the remaining weights are rescaled, so the score stays between 0 and 1. This is Gower's similarity coefficient (J. C. Gower, "A general coefficient of similarity and some of its properties", *Biometrics* 27(4), 1971), built for records that mix text-like, categorical and numeric attributes with gaps. It is also the problem statement's "fallback mechanism when certain features are unavailable", and stage 7 uses it for products without a photo.

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
| Exact search over every product | 22 ms per query at 30,000 products; the exact answer that stage 8's ANN index is measured against | Time grows with catalogue size |

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
| Module | `app.py` at the repo root, as in the problem statement's sample |
| Notebook | `notebooks/04_api.ipynb` |
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

From `notebooks/04_api.ipynb`, for A ("People Men's Printed Regular fit T-Shirt", the stage 3 example):

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
- **Effect.** 200 random products in `notebooks/04_api.ipynb`: 29.7 ms per request the first time, 5.6 ms the second. Over real HTTP: 38 ms and 2 ms.
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

`requirements.txt` now holds only what the image runs: fastapi, uvicorn, numpy, pandas, scikit-learn, pyarrow. `requirements-dev.txt` adds pytest, httpx (for FastAPI's test client) and matplotlib. torch, torchvision and pillow were removed: nothing imports them yet, and they would add over 1 GB to the image. Stages 6 and 7 later added `sentence-transformers` (which brings torch) and pillow to `requirements-dev.txt` only: they build the embeddings offline, and search only reads the saved numbers.

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
| `app.py` at the repo root | Matches the problem statement's sample and its `uvicorn app:app` | One module outside `similarity/`; it holds only the web layer |
| Return a plain list of IDs | The problem statement's signature is `-> List[str]` | No scores or names in the answer; adding them later changes the format |
| `num_similar` at most 100 | Bounds the work and the size of every answer; no page shows 100 recommendations | A caller wanting more gets 422 |
| Check for the ID before searching, instead of catching `KeyError` | A `KeyError` caused by a bug inside the search would otherwise be reported as "not found" | — |
| Plain 500 message | A traceback can reveal code and file paths; it goes to the log instead | The caller learns nothing about the cause |
| Load at startup, not on the first request | The first caller does not wait; `/health` answering means the server can really answer | Startup takes 2.8 s |
| Cache inside the process, not a shared one such as Redis | A miss costs only 22 ms; no extra service to run | One cache per container |
| `def`, not `async def`, for the endpoint | The search is numpy work, not waiting on a network. FastAPI runs a `def` endpoint in a thread pool, so one slow request does not block others. | — |

### Open issues

- **The image could be smaller.** Removing pip, setuptools and the packages' test folders in the build stage would save about 110 MB. Saving the table in a format pandas reads without pyarrow would save about 160 MB more, but Parquet keeps the types and lists (see stage 1). The 206 MB compressed size is downloaded once per machine, so this is not urgent.
- **Python 3.10 reached end of life in October 2026** and gets no more security fixes. It is kept because the problem statement's sample uses it. Moving to 3.12+ is a one-line change in the Dockerfile and would also allow current releases (pyarrow 26+, NumPy 2.3+).
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
| Data | Everything is in `evals/eval_results.xlsx`, with no separate files. The **Queries** sheet holds the 20 test products. The **Labels** sheet holds the 3,841 labels from the LLM judge, with the ground truth and where each candidate came from (`source`: shortlist or sweep). |
| Run | `python -m evals.run_eval` rebuilds `evals/eval_results.xlsx` and prints the same numbers (about 10 s). Close the workbook in Excel first. |
| Tests | `tests/test_evals.py` (7 tests; 95 across all stages) |
| Notebook | None. The run produces the Excel workbook. |

### Method

1. **Pick 20 test products** (queries). 12 are random (fixed seed): 10 spread over the 8 largest categories, plus 2 from any other category. 8 are hard cases chosen for a known difficulty (table below).
2. **Round one: a shortlist per test product.** 28 to 46 candidates, 736 in total, from four routes. They were shuffled so the judge could not tell which route found which:

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
4. **Label with an LLM as the judge.** The data has no labels, so there is no ground truth to borrow. An LLM judged every candidate from both rounds good or not good, with one rule (below).
5. **Pick the ground truth:** the 10 most similar good candidates from both rounds, best first.
6. **Score** the method's top 10 against the labels.

**Why the shortlist routes, and then a sweep:**
- **Why not just read the current method's results:** nobody can read 30,000 products for each test product, so the answer key has to come from a smaller set. If that set were the current method's own results, the "right answers" would be its own answers, like an answer key copied from the student's paper. It could never be caught missing a better product.
- **What the shortlist added:** the four routes put products the method misses in front of the judge. But the shortlist still only showed what those routes ranked highly. For the blouse fabric, every route leaned on its wrong category (kurtis), so not one fabric was read.
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
| **Ground truth found** | Of the judge's 10 closest products (the reference picks), how many the method also returned | 10 of 10 |
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

The **Methods** sheet gives every method in `evals/run_eval.py` one row with standard ranking metrics, so methods can be compared side by side. Each test product's 10 reference picks are the right answers, and every metric is averaged over the 20 test products:

| Metric | Meaning | `tfidf` |
|---|---|---:|
| **Recall@5** | Share of the 10 picks in the top 5 (at most 50%, since 5 slots cannot hold 10) | 24% |
| **Recall@10** | Share of the 10 picks in the top 10 | 32% |
| **MRR@10** | 1 ÷ rank of the first pick, averaged; 1.0 means a pick always comes first | 0.82 |
| **Good@10** | Share of the top 10 labelled good: the only metric using the good / not good labels | 90% |
| **ms per query** | Search time per test product (latency) | 22 |

Precision@5, hit rate and nDCG were tried and dropped to keep the table short. With exactly 10 picks per test product they repeat what recall and MRR already show.

**In short:** the first result is usually one of the closest products (MRR 0.82), but only about a third of the closest 10 make it into the top 10 (Recall@10 32%).

The metrics against the reference picks need no new labelling, so any new method gets them immediately. Only Good@10 needs its results labelled first. The numbers are computed by the script (`ranking_metrics` in `evals/run_eval.py`, tested against a hand-worked example) and written as values. Hovering over a column header shows its meaning.

### How to read the workbook

- **Methods:** one row per method with the ranking metrics above.
- **Summary:** one row per test product with good results, ground truth found and first good result for `tfidf` (and the `--alternative` method, if one is given), then the overall numbers. These are Excel formulas over Details.
- **Details:** 10 rows per test product. The ground truth sits beside the method's top 10, and each result is marked Good? and In ground truth? (green / red).
- **Queries, Labels:** the data. The yellow cells (`good`, `gt_rank`) are the ones to edit. A change made there is kept the next time `python -m evals.run_eval` runs.
- **Good** means a shopper would accept it as a similar alternative: same kind of product, same audience, similar style, price within about 3x.

Excel recalculated all 86 formulas with no errors, and every value matched the script's own Python calculation.

**Adding a method** (as stage 6 did for `minilm`):
1. Add it to `METHODS` in `evals/run_eval.py`: a description, a score and a search mode, the same two choices as the API's settings `SIMILARITY_METHOD` and `SEARCH_MODE`. A new score is a set of weights, added to `METHOD_WEIGHTS` in `similarity/search.py` and to `SIMILARITY_METHODS` in `similarity/config.py`.
2. Run `python -m evals.run_eval`. It gets a row on the Methods sheet. Add `--alternative <name>` to also show its results beside `tfidf` in Summary and Details. A method whose files are missing is skipped with a message.
3. Its results that were never labelled are added to the Labels sheet with an empty `good` cell, and the run warns about them. Fill in yes / no and run again; until then only Good@10 is incomplete.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| An LLM as the judge, run once in this session rather than through an API | The data has no labels. No API key or cost, and simpler. The labels are saved, so every method is scored on identical ground. | One judge, and an LLM, not a shopper. Worth having a person spot-check a few test products. |
| Shortlist from four routes, then a full-catalogue sweep | The sweep removes the dependence on what the routes rank highly. Its terms come from each product's own title, not from any method. | The terms are judgement. A matching product that uses none of the words is still missed. |
| Sweep tightened to about 150–200 products per test product | For common kinds (kurtas, sarees) hundreds of products are near-equal, so reading 1,400 would mostly reorder equals | The ground truth is the best of the closest 150–200, not of every product of that kind |
| Good/not good for every candidate, plus a ranked top 10 | A strict ID match alone would punish equally good near-copies | Two numbers to read instead of one |
| Price only as a 3x limit | Keeps "good" about what the product is | Price closeness is barely tested |
| Unlabelled results shown as "not labelled", with a warning | A new method's unseen results are neither right nor wrong until labelled | They must be labelled before its numbers are final |
| All data in the workbook, no separate files | One file to open, share and edit. The Labels sheet is read back on every run, so edits made in Excel are kept. | The labels live in a binary file: git cannot show which label changed. Excel must be closed while the script runs. |
| Ranking metrics against the reference picks, in their own sheet | Standard numbers (recall, MRR) that score any method straight away, with no new labelling; plus Good@10 and latency | Near-copies keep recall low even for a good method, so Good@10 is reported next to it |
| Methods metrics as values; Summary as formulas | The Methods sheet scores every registered method, including ones not shown in Details, so it cannot be formulas over Details | Editing a label in Excel updates Summary immediately, but Methods only after a rerun |
| Own folder `evals/`, no notebook | Evaluation stays apart from the service; one command produces the workbook | — |

### Open issues

- **One judge, an LLM,** labelling from text, not photos. The uncertain calls are listed above, under what counts as good, and can be corrected on the Labels sheet.
- **Good results is near its ceiling** for the current method (`tfidf`). Stage 6 improvements will show mainly in ground truth found and on the hard test products. With 20 test products, only large differences are meaningful.
- **The sweep's T-shirt pattern also matched "fit Shirt".** Some collared shirts were read (and labelled not good), and a few "...fit Shirt" titles were left out of the Q06 and Q08 sweeps. `evals/sweep.py` keeps the pattern as run, with a note.
- **Every result from the current method is labelled,** because its top 15 was one of the shortlist routes. Another method's results may need labelling first (stage 6 labelled 46 for `minilm`).
- **The current method is called `tfidf`** from stage 6 on, so the four methods have matching names.

## Stage 6: Title embeddings

| | |
|---|---|
| Module | `similarity/embeddings.py` (encode, save, load, rescale); the `text_embedding` score in `similarity/search.py` |
| Notebook | `notebooks/06_text_embeddings.ipynb` |
| Run once | `python -m similarity.embeddings`: 226 s on this laptop, capped at 4 CPU threads |
| Output | `data/processed/title_embeddings.npz`: 30,000 × 384 numbers (50 MB), with the uniq_ids and the random-pair score |
| Method | `minilm` on the Methods sheet |
| New dependency | `sentence-transformers==6.3.0` (brings torch), in `requirements-dev.txt` only |

Four methods are compared, each a row on the Methods sheet. They share the stage 3 search and differ only in their weights. This stage adds `minilm`; stage 7 adds the two photo methods.

| Attribute | `tfidf` (stage 3) | `minilm` | `tfidf_image` | `minilm_image` |
|---|---|---|---|---|
| Title, TF-IDF | 0.30 | — | 0.30 | — |
| Title, all-MiniLM-L6-v2 | — | 0.30 | — | 0.30 |
| Category, gender, price, brand, colour, weight | 0.20, 0.10, 0.15, 0.15, 0.05, 0.05 | same | same | same |
| Photo, CLIP ViT-B/32 | — | — | 0.20 | 0.20 |

### Method

1. **Encode once.** The pretrained model all-MiniLM-L6-v2 (Reimers & Gurevych, 2019; 23M parameters, downloaded once, about 90 MB) reads each product name as it is: the same text TF-IDF reads. It outputs 384 numbers per title, scaled to length 1. Nothing is trained. The name is the only descriptive text in the data: `meta_keywords` repeats it and `product_details` holds dates and ranks.
2. **Score.** At search time the title score is the query's numbers times every product's numbers (the cosine), rescaled as below. One multiplication over a 30,000 × 384 table.
3. **Combine.** It takes the TF-IDF title score's place at the same weight (0.30). Every other attribute and weight stays (`method_weights("minilm")` in `search.py`).
4. **Load only when used.** The numbers are read the first time a method uses them. The API's method (`tfidf`) never reads the file, so the Docker image needs neither the file nor torch.

### How it differs from TF-IDF

| | TF-IDF | MiniLM |
|---|---|---|
| Numbers per title | 7,412 word slots, about 8 filled | 384, all filled |
| Knowledge from | Counting words in our 30,000 titles | Pretraining on about 1 billion sentence pairs |
| Scores high when two titles… | Share words | Mean similar things |

Pairs scored by both (notebook):

| Title a | Title b | TF-IDF | MiniLM (rescaled) |
|---|---|---:|---:|
| Girls cotton frock | Girls cotton dress | 0.49 | 0.54 |
| Women's straight kurta | Women's straight kurti | 0.59 | 0.92 |
| Girls cotton frok | Girls cotton frock | 0.27 | 0.84 |
| Men's cotton T-shirt | Men's cotton tee | 0.36 | 0.72 |
| Woolen beanie | Winter knit cap | 0.00 | 0.12 |
| Girls cotton frock | Men's leather wallet | 0.00 | 0.00 |

With the title alone, TF-IDF's nearest titles to the Zadmus baby frock (Q14) are other Zadmus products, including boys' T-shirts. MiniLM's are frocks from other brands.

### Rescaling

Embedding cosines are rarely 0: two unrelated titles still share a language. Over 5,000 random pairs, the raw MiniLM score is above 0 for 99.8% of them, with a median of 0.31; TF-IDF is 0 for 63%. Left as is, every product would get a free title score of about 0.3. So each score is rescaled:

```
score = max(0, (cosine − b) / (1 − b))     b = 0.309, the median of 10,000 random pairs
```

A typical random pair then scores 0 and an identical title 1, the same shape as TF-IDF. After rescaling, 49% of random pairs score above 0. Stage 7 uses the same rescaling for photos.

### Labelling the new results

46 of `minilm`'s top-10 results had never been labelled. They were labelled with the stage 5 rules: 34 good, 12 not good.

Calls already made in stage 5 were applied unchanged:
- stitched kurtas are not fabric (Q13);
- a romper is not a frock;
- caps and snapbacks are not beanies;
- anarkali kurtas are not dresses;
- a tactical cap is not a newsboy cap.

Three calls were new:
- a plain adult beanie counts for the Marvel logo beanie;
- an animal-print boys' T-shirt counts for the character T-shirt;
- a woollen girls' frock counts as a frock.

The reference picks (ground truth) were not changed. Re-picking them after seeing a method's results would favour that method.

### Results

| Method | Recall@5 | Recall@10 | MRR@10 | Good@10 | ms per query |
|---|---:|---:|---:|---:|---:|
| `tfidf` | 24% | 32% | 0.82 | 90% | 26 |
| `minilm` | 24% | 31% | 0.71 | 88% | 24 |

Where they differ (reference picks found in the top 10, and good results, out of 10). The other 11 score the same; Q08 differs only in where its first pick lands.

| Test product | `tfidf` found / good | `minilm` found / good | Why |
|---|---|---|---|
| Q12 Jockey cotton brief, pack of 3 | 3 / 10 | **6** / 9 | Finds other brands' cotton briefs by meaning |
| Q13 Kalamkari blouse fabric | 1 / 1 | **2 / 2** | A second Kalamkari fabric now ranks 2nd |
| Q14 Baby girls' frock | 3 / 8 | **4** / 7 | Frocks from other brands |
| Q03 Plain regular-fit tee, Q10 A-line dress | 4, 5 | **5, 6** | One more pick each |
| Q18 "Pari Singh Orange Plain Women's Frok" | 3 / 10 | **0 / 8** | The title's meaning leans to ethnic wear, so anarkali kurtas come first. TF-IDF fell back on the category and found dresses. |
| Q16 Marvel Avengers logo beanie | 3 / 10 | **1 / 8** | "Marvel Avengers" outweighs "beanie", so Marvel caps rank high. TF-IDF weighs the rare word "beanie". |
| Q19 "OOMPH!" | 6 / 10 | **4** / 10 | A one-word brand title carries little meaning |
| Q04 Plain slim-fit tee | 5 / 10 | **3** / 10 | "Slim fit" and "regular fit" mean nearly the same to the model |

### Findings

- **MiniLM ties TF-IDF overall.** Recall and Good@10 are within one or two points, which is noise with 20 test products. MRR is lower (0.71 vs 0.82), because the first reference pick lands later on Q04, Q08, Q16 and Q18.
- **It helps where words differ but the meaning matches** (briefs, fabric, frocks from other brands) and **hurts where one exact word decides the product** (beanie, slim fit) or the title is mostly a brand.
- **The titles suit TF-IDF.** They are short (9 words on average) and keyword-like, and the other attributes carry 70% of the weight, so the title score moves the ranking less than it would alone.
- **Q11 (newsboy cap) and Q17 (toddler tee) still find no picks** with either title score.
- **Speed is the same:** both take about 25 ms per search (timings vary by about ±10 ms between runs on this laptop).
- **Not tried:** using both title scores together, TF-IDF for exact words and MiniLM for meaning.

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| all-MiniLM-L6-v2 | The standard small sentence transformer; fast on a CPU | Newer models (for example bge-small) score higher on public benchmarks; untested here |
| Same input as TF-IDF, the product name | Only the way the title becomes numbers changes, so the comparison is fair | — |
| Same weight as TF-IDF (0.30) | Only the title score changes | The best weight for MiniLM is untested |
| Rescale by the random-pair median | A typical random pair scores 0, like TF-IDF | About half of all products get a small positive title score |
| Encoded once, read on first use | The API needs no model or torch; a method that does not use the numbers never reads them | The file must be rebuilt when the products change (loading checks the uniq_ids and refuses a mismatch) |
| Encoding capped at 4 CPU threads | Keeps the load moderate on this laptop | Slower encoding (226 s) |
| Reference picks not re-picked | Avoids favouring the method just seen | The picks were made from text; fair for a text method, less so for photos (stage 7) |

### Open issues

- **20 test products** separate only large differences; most of the gaps above are one or two products.
- **One judge, an LLM,** labelling from text, as in stage 5.
- **Indian terms:** the model learned mostly from Western English text. It links kurta and kurti (0.92), but rare terms ("gota patti", brand names) carry little meaning for it.

## Stage 7: Image similarity (optional)

| | |
|---|---|
| Module | `similarity/images.py` (download, embed); the `image` score in `similarity/search.py` |
| Notebook | `notebooks/07_image_similarity.ipynb` |
| Run once, by hand | `python -m similarity.images download`, then `python -m similarity.images embed` |
| Photos | 29,861 of 30,000 (345 MB), **outside the repo** in `C:\Users\aswin\Downloads\Amazon Image Download` |
| Output | `data/processed/image_embeddings.npz` (65 MB): 512 numbers per photo, which products have a photo, and the random-pair score |
| Methods | `tfidf_image`, `minilm_image` on the Methods sheet |
| Dependency | pillow, plus `sentence-transformers` from stage 6, in `requirements-dev.txt` only |

**Standalone.** Nothing else runs these steps. Preprocessing, the Docker build, the API and the evaluation never download or encode a photo; the photo methods only read `image_embeddings.npz`, and the evaluation skips them with a message when it is missing.

**Outside the repo.** The repo sits in OneDrive, and 30,000 photos would sync to the cloud. The folder is the setting `IMAGE_DIR` (default: `Downloads/Amazon Image Download` in the home folder), and `--image-dir` changes it for one run.

### Method

1. **Which photo.** The first link in `medium`, the product's main photo at about 500 px tall. `small` is a 38 × 50 thumbnail, too small to show a print. `large` is a zoom version of up to 1,500 px: five times the download for nothing at the size the model looks at, and missing for 1,159 products.
2. **Download.** Eight at a time, with a 10 s timeout, two retries and a generic user agent. It would stop if the server refused 50 requests in a row; that never happened. Each photo is padded to a white square, so a tall photo is not cropped, then shrunk to 224 × 224 and saved as `<uniq_id>.jpg`. A photo already saved is skipped, so a stopped download carries on where it stopped.
3. **Encode.** CLIP ViT-B/32 (Radford et al., 2021), through sentence-transformers, turns each photo into 512 numbers, scaled to length 1. Nothing is trained. Work is saved every 1,000 photos, so a stopped run loses at most one chunk.
4. **Score.** The cosine of two photos' numbers, rescaled as in stage 6 with b = 0.661.
5. **Combine.** The photo is added on top of the other weights at 0.20. Weights are relative: with a photo, the title's share drops from 30% to 25% and the photo's is 17%.
6. **Fallback.** When either product has no photo, the photo score is skipped for that pair, like any missing attribute (Gower). A product without a photo therefore ranks exactly as under the text-only method: the notebook shows the same top 10, and a test checks it.

**Why CLIP, not a CNN.** The brief suggests a pretrained CNN (ResNet, EfficientNet). CLIP is a vision transformer. It was chosen for what it learned from, not its architecture: 400 million images with captions, many of them shopping-style, against ImageNet's 1,000 object classes with almost no clothing (no saree, kurta or frock). CLIP also runs through the same library as stage 6.

### Photos

| | Count |
|---|---:|
| Downloaded | 29,861 |
| Link no longer exists (HTTP 404) | 137 |
| No link in the data | 2 |
| **Without a photo** | **139** |

The download took about 13 minutes over four runs. The laptop switched off three times during it, and also when idle, so the download was not the cause. Each run carried on where the last stopped, and every saved photo was checked to open. Encoding took 26 minutes for the first 29,000 photos; the last 861 took 25 more, most likely while the laptop slept.

**No placeholder pictures.** An "image not available" picture shared by many products would make them all match each other. The largest group of identical photos is 6: one vest in several sizes. 564 photos are shared by 2 to 6 products, each the same item listed more than once.

### Rescaling

Two product photos always look somewhat alike: clothes, centred, on a plain background. Over 5,000 random pairs the raw cosine is never below 0.36, with a median of 0.66. Rescaled with b = 0.661, the median pair scores 0 and the 90th percentile 0.39.

### Labelling the new results

44 results of the photo methods had never been labelled. They were labelled **while looking at the photos**: one contact sheet per test product showing its photo and each result. 39 are good and 5 not good:
- Q11 (newsboy cap): a baseball cap, a sun hat and a fedora.
- Q16 (beanie): a trucker cap.
- Q17 (toddler T-shirt): a Monte Carlo "T-shirt" whose photo shows a sweatshirt.

The reference picks were not changed.

### Results

| Method | Recall@5 | Recall@10 | MRR@10 | Good@10 | ms per query |
|---|---:|---:|---:|---:|---:|
| `tfidf` | 24% | 32% | **0.82** | 90% | 42 |
| `minilm` | 24% | 31% | 0.71 | 88% | 40 |
| `tfidf_image` | 23% | 31% | 0.73 | 89% | 43 |
| `minilm_image` | 22% | 30% | 0.71 | 90% | 43 |

The timings come from one run in which the laptop was slower than during stage 6 (the same `tfidf` search took 26 ms there). The photo adds about 2 to 3 ms: one more multiplication, over a 30,000 × 512 table.

Where the photo changes the result, compared with the same method without it (reference picks found / good results, out of 10):

| Test product | Without photo | With photo | What happened |
|---|---|---|---|
| Q20 Captain America by Kidsville boys' tee | 3 / 10 (`tfidf`) | **5** / 9 | Other Kidsville character tees (Iron Man) look alike |
| Q13 Kalamkari blouse fabric | 1 / 1 (`tfidf`), 2 / 2 (`minilm`) | **2 / 2**, **2 / 3** | A Kalamkari and a Bagru-print fabric: photos of fabric look like fabric |
| Q19 "OOMPH!" | 6 / 10, 4 / 10 | **7** / 10, **5** / 10 | Sarees look like sarees; the title says nothing |
| Q14 Baby girls' frock | 3 / 8, 4 / 7 | 2 / **9**, 3 / **9** | Striped dresses like the query: more good results, fewer of the reference picks |
| Q09 Max boys' tee | first pick at rank 3 | **rank 1** | |
| Q04 Plain slim-fit tee | 5 / 10, 3 / 10 | **2** / 10, **1** / 10 | The title says "Plain", but the photo shows a printed tee. The photo methods find printed tees (all labelled good); the reference picks, made from the title, are plain. |
| Q18 Orange plain women's frock | 3 / 10 (`tfidf`) | **0** / 10 | Ten good short dresses that look alike, none of them a reference pick |
| Q16 Marvel logo beanie | 3 / 10 (`tfidf`) | 3 / **8**, first pick at rank 4 | Two caps get in; the photo of a black beanie resembles a black cap |
| Q05 Silk saree | first pick at rank 1 | rank 2 | |

### Findings

- **No method beats `tfidf` on these numbers.** All four find 30% to 32% of the reference picks and 88% to 90% good results; `tfidf` puts a reference pick first most often (MRR 0.82). With 20 test products, the other differences are within noise.
- **The photo changes which products are found more than how many good ones.** It helps where the title says little or the look is the point (OOMPH!, the fabric, character tees). It hurts where the photo and title disagree (Q04) or where looks cross product kinds (a black beanie and a black cap).
- **The yardstick favours text.** The reference picks were chosen from titles. Q04 and Q18 show the photo methods returning ten good products that score 0 to 2 on recall. A fair test of the photo would need reference picks chosen while looking at the photos.
- **The data itself disagrees with itself:** Q04's title says "Plain" for a printed T-shirt. Neither title nor photo is always right.
- **Q11 (newsboy cap) and Q17 (toddler tee) still find no pick** under any method.
- **The API keeps `tfidf`:** best MRR, the same Good@10, and no model, photo folder or 65 MB file to ship. The photo methods remain available through the setting `SIMILARITY_METHOD` (see Settings).

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| The main `medium` photo only | Twice the size CLIP needs; one front view per product | Back views and close-ups are not used |
| Pad to a white square, 224 × 224 | Nothing cropped; the size CLIP looks at | Fine print details shrink |
| CLIP ViT-B/32 instead of a CNN | Learned from images with captions; the same library as stage 6 | Not the CNN the brief names; CLIP is known to react to text in photos (logos, watermarks) |
| Photo weight 0.20, added on top | A noticeable but minority share (17%) | Only one weight was tried |
| Missing photo skipped, not scored 0 | The brief's fallback; a product without a photo ranks as under the text-only method | — |
| Photos outside the repo | The repo is in OneDrive | Another machine needs the photos in its own home folder, or `IMAGE_DIR` set |
| Download and encoding save as they go | The laptop switched off several times; no work was lost | A few extra files during the run |
| New results labelled from the photos | Style and kind are clearer in a photo | The reference picks are still text-based |
| Not in the API | Nothing beat `tfidf`; the image stays small | — |

### Open issues

- **Reference picks chosen from text** under-credit the photo methods (Q04, Q18).
- **Only one photo weight (0.20)** was tried. Rows at 0.10 and 0.30 would show how sensitive the results are.
- **The photos live outside the repo.** Another machine needs them downloaded again, or `IMAGE_DIR` pointed at a copy.
- **139 products have no photo** and are always judged on text and attributes alone.

## Stage 8: ANN index

| | |
|---|---|
| Module | `similarity/ann.py` (the product vector, the index, the two-step search); `evals/ann_benchmark.py` (the measurements) |
| Notebook | `notebooks/08_ann.ipynb` |
| Run once, by hand | `python -m similarity.ann build`, after stage 6 (about a minute) |
| Output | `data/processed/product_hnsw.faiss` (162 MB): one product vector of 1,214 numbers per product, and the HNSW graph that links them |
| Measure | `python -m evals.ann_benchmark measure` (the 30,000 real products), `python -m evals.ann_benchmark scale` (1,000,000 made-up vectors) |
| Methods | `tfidf_ann` on the Methods sheet |
| Dependency | faiss-cpu 1.16.0, in `requirements-dev.txt` only |
| API | Exact search by default; `SEARCH_MODE=ann` switches it to the two-step search (see Settings) |

**What Part 3 asks:** a vector search for large datasets, the paper behind it, and why it was chosen. The problem statement's suggested approaches name three ways to make search cheaper. This stage covers ANN indexing, and stage 4 already caches each product's top 100. Dimensionality reduction (PCA) was left out by decision; see Open issues.

### How HNSW works

**Exact search** compares the query with every product. Its time grows in step with the catalogue: 22 ms here, and about 1 s at a million products (measured below).

**HNSW** (Hierarchical Navigable Small World graphs, Malkov & Yashunin, 2016; IEEE TPAMI 2018) avoids comparing with everything:

1. **A graph.** Each product is linked to up to M of its nearest products.
2. **A walk.** A search starts at a fixed entry product and moves to whichever linked product is closest to the query. It repeats until no link gets closer. It keeps a list of the best `efSearch` products seen so far, so it can back out of a dead end.
3. **Layers** (the "hierarchical"). Every product sits on the bottom layer, and fewer and fewer also sit on the layers above. The top layers work like motorways: a search crosses the catalogue in a few long jumps, then drops a layer and walks the local streets. So its time grows with the logarithm of the catalogue size, not with the size itself.

Our index has 3 layers: all 30,000 products on the bottom layer, 479 also on the middle one, and 7 on the top.

| Setting | Value | What it does |
|---|---|---|
| M | 64 | Links per product on each layer. More links: fewer misses, more memory. |
| efConstruction | 500 | How many candidates are weighed when a product is linked in. Larger: a better graph, a slower build. |
| efSearch | 64, raised to the shortlist size (1,000) | How many candidates a search keeps. Larger: better recall, slower search. A search keeps at least as many as it returns, so for the shortlist this is always 1,000; it is not a setting. |

The index is built with one thread, so the same vectors always give the same graph and the same results. The saved file is checked against the current product vectors when it is read. If the products or the title embeddings have changed, it refuses to load and asks for a rebuild.

### Why HNSW

| Option | How it narrows the search | Why not chosen |
|---|---|---|
| Exact search | It doesn't | The yardstick: always right, but time grows with the catalogue |
| IVF (FAISS) | Groups the vectors into k-means clusters and scans only the `nprobe` clusters nearest the query | Needs a training step and a cluster count to tune; misses neighbours that fall just across a cluster border |
| Product quantization (Jégou, Douze & Schmid, 2011) | Compresses each vector to a few bytes | Saves memory, which 30,000 products don't need, and costs recall. It is the step to add at hundreds of millions. |
| Annoy (Spotify) | Many random-split trees | A new product means rebuilding the trees; slower than HNSW at the same recall in ANN-Benchmarks |
| **HNSW** | **A graph walk** | **Chosen** |

Why HNSW:
- **Speed at high recall.** It is among the fastest methods at high recall in ANN-Benchmarks (Aumüller, Bernhardsson & Faithfull, *Information Systems*, 2020).
- **No training.** Products are added one at a time, so a new product doesn't mean a rebuild.
- **Three settings**, each with a clear meaning.
- **It ships in FAISS** (Johnson, Douze & Jégou, "Billion-scale similarity search with GPUs", 2017). FAISS also has IVF and product quantization for when a catalogue outgrows memory.

The problem statement's "available on Hugging Face" fits too: the `datasets` library attaches a FAISS index to a dataset (`add_faiss_index`). It is the same engine; FAISS is used directly here because our vectors live in numpy, not in a `Dataset`.

The costs:
- **Memory:** every vector plus its links.
- **Build time.**
- **No removal:** FAISS's HNSW cannot remove a product.
- **A few products can end up unreachable** (below).

### Our score is not one vector, so the search has two steps

The API's score mixes seven attributes and skips the ones a pair lacks (Gower, stage 3). An index needs one list of numbers per product, compared by a single dot product. So:

1. **Shortlist:** HNSW returns the 1,000 products whose vectors are closest to the query's.
2. **Re-rank:** only those 1,000 are scored with the usual weighted score, by the same `search.rank` code, applied to just those rows (`Features.take`).

A shortlisted product gets exactly its usual score. So the two steps can only go wrong in one way: a product that belongs in the top 10 is left off the shortlist.

### Which vector to index

**Title only (first try).** The plan was to index the MiniLM title vectors of stage 6. HNSW found the nearest titles almost perfectly (99.7% in 0.2 ms), but the final top 10 was often wrong. The title is only 30% of the API's score. For the frock test product (Q14), exact search's places 5 to 10 were girls' dresses from Biba and Cherokee: the same category, gender and price range, but their titles ranked 7,939th to 20,358th by meaning. No title-only shortlist could contain them.

**One product vector (final).** Each attribute becomes a block of numbers. Each block has length 1, so the dot product of two blocks is that attribute's similarity. Each block is multiplied by the square root of its weight, so the dot product of two whole vectors is the weighted sum.

| Block | Numbers | How | Weight |
|---|---:|---|---:|
| Title | 384 | The MiniLM numbers of stage 6, standing in for TF-IDF | 0.30 |
| Category | 520 | A 1 in the slot of the child category, or of the browsenode when there is none | 0.20 |
| Gender | 4 | A 1 in the slot of the gender | 0.10 |
| Brand | 256 | A 1 in one of 256 slots, by brand code (hashing): two of the 6,336 brands share a slot 1 time in 256 | 0.15 |
| Price | 50 | Overlapping bands on a log scale: a price twice another scores 0.5, the same as the API's smaller ÷ larger | 0.15 |
| **Total** | **1,214** | | |

Colour and weight are left out: they count 5% each and are missing for about 80% of products. A missing attribute is a block of 0s. Finally each vector is scaled to length 1, so the other blocks of a product with gaps count for more. That is close to the API skipping what is missing, though not equal to it: Gower divides by the weights *both* products have.

Recall@10 against exact search over 500 random products, with exact shortlists (no HNSW yet), so only the vector is being tested:

| Vector | Shortlist 250 | 500 | 1,000 |
|---|---:|---:|---:|
| Title only | 75.0% | 80.8% | 86.2% |
| + category, gender, brand | — | 90.4% | — |
| + price | 89.7% | 93.7% | 96.1% |
| **Scaled to length 1 (final)** | **92.8%** | **95.6%** | **97.2%** |
| TF-IDF title instead of MiniLM (not indexable, see below) | 94.8% | 96.4% | — |

**What the rest of the gap comes from:**
- **The title.** The API scores titles with TF-IDF, and the vector uses MiniLM. TF-IDF has 7,412 mostly empty numbers; HNSW needs a short, dense list, which would mean PCA first.
- **Gower's skipping of missing values,** which a fixed vector can only approximate.

### Unreachable products

The first build used M = 32 and efConstruction = 200. In the evaluation, `tfidf_ann` then lost the rank-1 reference pick of Q06 and of Q13, and its MRR fell from 0.82 to 0.72. Both picks were among the query's 2 closest vectors, yet a search 1,000 deep did not return them. Searching with each product's own vector showed **41 products (0.14%) that could not be found at all**.

HNSW keeps at most M links per product. When a product collects too many links, it drops some, so a product can end up with no link pointing to it. Identical vectors were not the cause: there are 1,466 groups of them, and indexing each only once still left 41 unreachable.

| M | efConstruction | Unreachable at efSearch 64 | at 1,000 (the shortlist) |
|---:|---:|---:|---:|
| 32 | 200 | 41 | 11 |
| 32 | 500 | 15 | 4 |
| 64 | 500 (final) | 10 | **0** |
| 128 | 200 | 26 | 0 |

With M = 64 and efConstruction = 500, every product is found at the shortlist depth. The cost: the index grows from 154 to 162 MB, and the build from 24 to 63 s.

### Results: the 30,000 real products

500 random products. Step 1 alone, finding the 10 nearest product vectors:

| Search | Recall@10 | ms per query |
|---|---:|---:|
| Exact | 100% | 5.9 |
| HNSW, efSearch 10 | 98.3% | 0.12 |
| HNSW, efSearch 32 | 99.8% | 0.25 |
| HNSW, efSearch 64 | 99.8% | 0.41 |
| HNSW, efSearch 256 | 99.8% | 1.11 |

Both steps, compared with the API's top 10:

| Search | Recall@10 | Same top 10 | ms per query |
|---|---:|---:|---:|
| Exact (the API) | 100% | 100% | 22.3 |
| Shortlist 100 + re-rank | 88.0% | 61.0% | 1.7 |
| Shortlist 250 + re-rank | 92.6% | 73.8% | 2.5 |
| Shortlist 500 + re-rank | 95.5% | 81.2% | 3.8 |
| **Shortlist 1,000 + re-rank (chosen)** | **97.2%** | **87.4%** | **6.5** |
| Shortlist 2,000 + re-rank | 98.4% | 91.4% | 13.1 |

**How recall is counted:** a result counts as found when it scores at least as well as exact search's 10th result. Many products share a title, so several can tie with the 10th, and returning a different one of them is not a miss (as in ANN-Benchmarks). "Same top 10" is the strict measure: the identical ten products, in the same order.

With the shortlist of 1,000, the HNSW shortlist reaches the same recall as an exact one (97.2%), so the index itself adds no error; all of it comes from the vector. Re-ranking costs about 2.2 ms for 1,000 products, the same at any catalogue size.

### Results: evaluation

| Method | Recall@5 | Recall@10 | MRR@10 | Good@10 | ms per query |
|---|---:|---:|---:|---:|---:|
| `tfidf` (the API) | 24% | 32% | 0.82 | 90% | 21.7 |
| `tfidf_ann` | 24% | 32% | 0.82 | 90% | 8.1 |

Only 2 of the 20 test products get a different top 10, and both score the same:
- **Q04 (Flying Machine tee):** places 7 to 10 are reordered among the same good Flying Machine T-shirts.
- **Q05 (Banarasi silk saree):** place 10 is a different good saree.

Every result had already been labelled, so no new labels were needed.

### Scale test: 1,000,000 vectors

The catalogue has only 30,000 products, so a million vectors were made up:
- **How:** copies of real title vectors, each number nudged by random noise (σ = 0.02), then scaled back to length 1. A copy keeps a cosine of about 0.93 with its original. Purely random vectors would not do: in hundreds of dimensions, every pair is about equally far apart, which no catalogue looks like.
- **Size:** the copies have the 384 title numbers, not the 1,214 of the product vector, which would need 4.9 GB.
- **Build:** the index was built with all 16 threads, since only its time matters here.

| Catalogue | 30,000 (real) | 200,000 | 1,000,000 |
|---|---:|---:|---:|
| Exact search on the vectors alone | 2.0 ms | 11.9 ms | 55.4 ms |
| HNSW, efSearch 16: recall@10, time | — | 99.5%, 0.12 ms | 98.2%, 0.12 ms |
| HNSW, efSearch 64: recall@10, time | — | 99.9%, 0.32 ms | 100%, 0.28 ms |
| Index build (16 threads) | — | 150 s | 708 s |

How the API's two options grow (ms per query):

| Catalogue | Exact search, all seven attributes (today's API code) | Two steps: shortlist of 1,000 + re-rank |
|---|---:|---:|
| 30,000 | 22 | 6.2 (measured) |
| 200,000 | 154 | about 6.5 (4.3 + 2.2) |
| 1,000,000 | 1,109 | about 6.1 (3.9 + 2.2) |

Exact search with all seven attributes was timed with the real products repeated to fill each catalogue size; it scores and sorts every row, as the API does. Its time grows a little faster than the catalogue (sorting). The two-step column adds the measured shortlist time to the 2.2 ms that re-ranking 1,000 products takes at any size. Beyond 30,000 its shortlist was timed on 384-number vectors, so with the full 1,214-number product vector it would be somewhat slower; at 30,000 the product-vector shortlist takes 3.9 ms.

### Findings

- **HNSW itself is close to perfect, and its time hardly grows.** At a million vectors it finds all 10 nearest at efSearch 64 in 0.28 ms, against 55 ms for exact search on the same vectors: about 200 times faster. From 200,000 to 1,000,000 vectors its time stayed the same while exact search grew 4.7 times. That is the logarithmic growth the layers give.
- **The hard part was the score, not the index.** The API's score is not one vector. A title-only shortlist recovered 86% of the API's top 10 at 1,000; the product vector recovers 97%.
- **On the 20 test products, `tfidf_ann` scores exactly like `tfidf`** (MRR 0.82, Good@10 90%), in 8 ms instead of 22.
- **At 30,000 products the gain is about 15 ms, which nobody notices.** A repeated product is answered from the cache in 2 ms. At a million products, exact search takes 1.1 s, against about 6 ms for the two steps.
- **Exact search passes 100 ms at about 150,000 products** (154 ms at 200,000). That is roughly where switching the API to the two-step search starts to pay.
- **HNSW can leave products unreachable.** 41 were, at M = 32. It is worth checking in any HNSW deployment; here M = 64 and efConstruction = 500 fixed it.
- **The photo is not in the index.** The index copies the API's score, and the API has no photo score (stage 7).

### Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| HNSW, from FAISS | Fast at high recall, no training, three clear settings; FAISS also holds IVF and PQ for later | FAISS's HNSW cannot remove a product; memory for the links |
| One product vector, not titles alone | Titles alone recovered 81% of the API's top 10 at a shortlist of 500; the product vector 96% | 1,214 numbers per product; an approximation of Gower |
| MiniLM for the title block | Short and dense, already built in stage 6 | Not the API's TF-IDF: about 1 point of recall |
| Brand hashed into 256 slots | 6,336 brands would need 6,336 numbers | Two brands share a slot 1 time in 256 |
| Colour and weight left out | 5% each, missing for about 80% of products | Re-ranking still uses them |
| Shortlist then exact re-rank | The API's own score decides the order; a shortlisted product scores exactly as before | Re-ranking costs 2.2 ms per query |
| Shortlist of 1,000 | 97% recall, 87% identical top 10, about 3 times faster than exact at 30,000 | 2.8% of the top 10 is a different, slightly lower-scoring product |
| M = 64, efConstruction = 500 | Every product reachable at the shortlist depth | 162 MB, 63 s build |
| Built with one thread, file checked on load | The same results on every run; a stale index refuses to load | A slower build |
| The API's default stays exact search | 22 ms (2 ms from the cache) is fast enough at 30,000; no faiss, index file or build step in the image | ANN is one setting away (`SEARCH_MODE=ann`), but the Docker image would need faiss-cpu and the index |
| No PCA | Decided before the stage | The TF-IDF title cannot go into the index |

### Open issues

- **PCA was not done.** It would let the TF-IDF title into the index (about 1 point of recall) and shorten the vectors, which matters at a million products: 1,214 numbers × 4 bytes × 1,000,000 = 4.9 GB.
- **The scale test uses made-up vectors.** Real neighbourhoods are more varied, so recall at a million real products may be lower.
- **Re-ranking copies the shortlisted rows** (`Features.take`) on every query. That is fast enough here, but a server would score the rows in place.
- **The index must be rebuilt** when the products or the title embeddings change. A stale file refuses to load, but nothing rebuilds it automatically.
- **The photo is not in the product vector.** It copies the API's score, which has no photo. A CLIP block, or a second index for photos whose shortlist is merged with this one, is the next step if photos are used.
- **10 products are still unreachable at efSearch 64.** They are only found because the shortlist search goes 1,000 deep. A small shortlist would need a check like the one in the notebook.
