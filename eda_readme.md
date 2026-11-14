# EDA: Amazon fashion sample

Profile of all 30,000 products in

`data/archive/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`

`eda.ipynb` is the raw pass: one column at a time, values first, then a count. `eda_learning.ipynb` runs the same cells, with a note above each one on what the call returns. Open either notebook from the repo root and run the load cell before the later cells.

## Dataset

| | |
|---|---|
| Rows | 30,000 |
| Columns | 33 |
| Distinct `uniq_id` | 30,000 |
| Distinct `asin` | 29,529 |
| Repeated `product_name` | 7,576 |

`uniq_id` is the product key. `asin` repeats, and many titles repeat, so neither can identify a row.

## Missing values

`weight`, `rating`, and `product_name` are present on every row. `weight` is present as a field, but most of its values are a sentinel (see below). The chart is `eda_output/missingness.png`.

| Column | Missing | Percent |
|---|---:|---:|
| `colour` | 23,971 | 79.9% |
| `discount_percentage` | 15,376 | 51.3% |
| `brand` | 8,143 | 27.1% |
| `other_items_customers_buy` | 5,637 | 18.8% |
| `sales_rank_in_child_category` | 5,149 | 17.2% |
| `sales_rank_in_parent_category` | 4,503 | 15.0% |
| `sales_price` | 2,890 | 9.6% |
| `product_details__k_v_pairs` | 1,183 | 3.9% |
| `large` (image URL) | 1,159 | 3.9% |
| `browsenode` | 520 | 1.7% |
| `image_urls__small`, `medium` | 2 | 0.0% |

These columns are almost empty and are not similarity features: `name_of_author_for_books`, `formats___editions`, `no__of_offers`, `no__of_sellers`, `technical_details__k_v_pairs`, `left_in_stock`, `no__of_reviews`, `seller_name`, `seller_id`.

## Price

`sales_price` is text and parses to a number for 27,110 rows. Nothing is zero or negative. Unparsed or missing: 2,890.

| | Rupees |
|---|---:|
| Min | 39 |
| 5th percentile | 219 |
| 25th percentile | 379 |
| Median | 590 |
| 75th percentile | 899 |
| 95th percentile | 2,877 |
| 99th percentile | 5,220 |
| Max | 9,988 |

The distribution is skewed, so price is compared on a log scale. Chart: `eda_output/price_log10.png`. There is no separate `price` column.

## Rating

Rating parses for all 30,000 rows. Min 1.0, median 4.0, max 5.0. The mass is at the top: 3,861 products are exactly 4.0 and 7,450 are exactly 5.0. Chart: `eda_output/rating.png`.

Rating can break ties. It is a weak similarity signal because so many products share the same value.

## Weight

| | Rows | Percent |
|---|---:|---:|
| Sentinel `999999999` | 23,745 | 79.1% |
| Parsed to grams | 6,254 | 20.8% |
| Unparsed text | 1 | — |

Parsed weights run from 1.2 g to 4,540 g, median 231 g. Chart: `eda_output/weight_grams.png`. The sentinel and the one row whose weight is a sentence are missing values. Real weights look like `200 g` or `99.8 g` and have to be parsed to grams.

## Brand and colour

| | Present | Distinct values |
|---|---:|---:|
| Brand | 21,857 | 6,458 |
| Colour | 6,029 | 4,757 |
| Both | 4,791 | — |

Largest brands: Max (504), Generic (245), BIBA (205), Mothercare (156), Campus Sutra (150). `Generic` is a placeholder and should be treated as missing.

Every non-empty `colour` value is a pipe-separated list of at least two colours, for example `Black|White` (79), `Black|Blue` (45), `Black|Red` (34). The shortest list has 2 colours; many list 3 or more. Matching the raw string almost never hits. Colour similarity has to compare the set of colours. Another 7,474 rows have an empty colour column but a colour word in `product_name`.

## Category

| | Present | Distinct values |
|---|---:|---:|
| `browsenode` | 29,480 | 302 |
| Child category | 24,851 | 218 |
| Parent category | 25,497 | 19 |

The parent category is `ClothingAccessories` for 24,893 of those 25,497 rows, so it barely separates products. The child category does. Largest child categories:

| Child category | Products |
|---|---:|
| WomensKurtasKurtis | 3,534 |
| MensT_Shirts | 3,385 |
| WomensSarees | 1,808 |
| MensFormalShirts | 1,060 |
| GirlsDressesJumpsuits | 992 |
| MensCasualShirts | 866 |
| BoysT_Shirts | 856 |
| WomensDresses | 759 |

Chart: `eda_output/child_categories.png`. The names come from the keys of `sales_rank_in_child_category`.

## Text and images

`product_name` length: min 4, median 50, max 500 characters. It is the only description present on every row.

| Image field | Present | Percent |
|---|---:|---:|
| `image_urls__small` | 29,998 | 100.0% |
| `medium` | 29,998 | 100.0% |
| `large` | 28,841 | 96.1% |

The later image model should download the `medium` URL. Several URLs are packed into one field, separated by `|`.

## Other flags

| Field | What it contains |
|---|---|
| `amazon_prime__y_or_n` | N 16,500, Y 13,500 |
| `best_seller_tag__y_or_n` | N 29,987, Y 13 |
| `delivery_type` | fulfilled by merchant 26,382, by Amazon 3,618 |
| `discount_percentage` | Text such as `50%`. After stripping `%`: 14,624 parsed, median 54, range 1 to 99 |

Best-seller is almost constant. Discount says how hard the item is marked down, not what the item is. Prime and delivery type describe fulfillment. None of these belong in the similarity score.

## What this means for the model

- Build the score so a missing brand, colour, price, or weight is left out of that pair, then renormalize the remaining weights.
- Let `product_name` and child category carry most queries. Parent category does not.
- Compare price on a log scale. Use rating as a tie-break.
- Treat `Generic` and weight `999999999` as missing.
- Compare colour as a set overlap, and only on the 6,029 rows that have the column.
- Leave out best-seller, discount, prime, delivery type, seller fields, and `other_items_customers_buy`.
- Image URLs already exist for almost every product. The visual model can use the `medium` URL when that step is added.
