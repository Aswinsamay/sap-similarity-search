"""Every setting that can change without editing code, in one place.

Each setting is read, in this order, from:
    1. an environment variable of the same name (`docker run -e`, the Kubernetes ConfigMap)
    2. the .env file in the repo root (copy .env.example to .env and edit it)
    3. the default below
Settings are read once, when Python starts: restart the API, or rerun the command, after a change.

The .env format is the one `docker run --env-file` reads: KEY=VALUE, one per line, with
comments on lines of their own (a # after a value is part of the value). SIMILARITY_ENV_FILE
names a different file; the tests set it to nothing, so they always run with the defaults.

Rules that describe the data (colour spellings, the weight sentinel, ...) stay in their
modules: they are facts found in the data, not choices to tune.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = os.environ.get("SIMILARITY_ENV_FILE", str(ROOT / ".env"))

SIMILARITY_METHODS = ("tfidf", "minilm", "tfidf_image", "minilm_image")
SEARCH_MODES = ("exact", "ann")


def read_env_file(path: str | Path | None) -> dict[str, str]:
    """The KEY=VALUE lines of a file; blank lines and lines starting with # are skipped.
    A missing file, or no path, gives nothing."""
    if not path or not Path(path).is_file():
        return {}
    values = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"{path}, line {number}: expected KEY=VALUE, got {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]   # "quoted" or 'quoted' -> quoted
        values[key.strip()] = value
    return values


_FILE = read_env_file(ENV_FILE)
DEFAULTS: dict[str, object] = {}   # every setting and its default, for .env.example and its test


def path(raw: str) -> Path:
    """A file or folder; ~ means the user's home folder."""
    return Path(raw).expanduser()


def setting(name: str, default, kind=str):
    """The value of one setting: environment, then .env, then the default."""
    DEFAULTS[name] = default
    raw = os.environ.get(name, _FILE.get(name))
    if raw is None or raw.strip() == "":
        return default
    try:
        return kind(raw.strip())
    except ValueError:
        raise ValueError(f"setting {name}={raw!r} is not a valid {kind.__name__}") from None


def choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = setting(name, default)
    if value not in allowed:
        raise ValueError(f"setting {name}={value!r}: choose one of {', '.join(allowed)}")
    return value


def at_least(name: str, default, minimum, kind=int):
    value = setting(name, default, kind)
    if value < minimum:
        raise ValueError(f"setting {name}={value}: must be at least {minimum}")
    return value


# --- the API: what it searches with ---------------------------------------------------------
SIMILARITY_METHOD = choice("SIMILARITY_METHOD", "tfidf", SIMILARITY_METHODS)   # which score
SEARCH_MODE = choice("SEARCH_MODE", "exact", SEARCH_MODES)                     # how the products are found
MAX_SIMILAR = at_least("MAX_SIMILAR", 100, 1)    # the most a caller may ask for; also how many are cached
CACHE_SIZE = at_least("CACHE_SIZE", 1024, 0)     # products whose results the API keeps in memory

# --- the score: how much each attribute counts (stages 3 and 7) ----------------------------
WEIGHT_TEXT = at_least("WEIGHT_TEXT", 0.30, 0, float)
WEIGHT_CATEGORY = at_least("WEIGHT_CATEGORY", 0.20, 0, float)
WEIGHT_GENDER = at_least("WEIGHT_GENDER", 0.10, 0, float)
WEIGHT_PRICE = at_least("WEIGHT_PRICE", 0.15, 0, float)
WEIGHT_BRAND = at_least("WEIGHT_BRAND", 0.15, 0, float)
WEIGHT_COLOUR = at_least("WEIGHT_COLOUR", 0.05, 0, float)
WEIGHT_WEIGHT = at_least("WEIGHT_WEIGHT", 0.05, 0, float)
WEIGHT_IMAGE = at_least("WEIGHT_IMAGE", 0.20, 0, float)      # added on top in the *_image methods
SCORE_DECIMALS = at_least("SCORE_DECIMALS", 3, 0)            # closer scores tie, and the rating decides

# --- ANN search (stage 8) -----------------------------------------------------------------
ANN_CANDIDATES = at_least("ANN_CANDIDATES", 1000, 10)        # shortlisted by HNSW, then scored exactly
ANN_M = at_least("ANN_M", 64, 2)                             # links per product (index build)
ANN_EF_CONSTRUCTION = at_least("ANN_EF_CONSTRUCTION", 500, 1)   # candidates weighed per link (index build)

# --- files ----------------------------------------------------------------------------------
RAW_DATA_PATH = setting("RAW_DATA_PATH", path(
    "data/archive/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"), path)
PRODUCTS_PATH = setting("PRODUCTS_PATH", path("data/processed/products.parquet"), path)
TITLE_EMBEDDINGS_PATH = setting("TITLE_EMBEDDINGS_PATH", path("data/processed/title_embeddings.npz"), path)
IMAGE_EMBEDDINGS_PATH = setting("IMAGE_EMBEDDINGS_PATH", path("data/processed/image_embeddings.npz"), path)
ANN_INDEX_PATH = setting("ANN_INDEX_PATH", path("data/processed/product_hnsw.faiss"), path)
IMAGE_DIR = setting("IMAGE_DIR", path("~/Downloads/Amazon Image Download"), path)   # the photos, outside the repo

# --- the offline steps (stages 6 and 7) ------------------------------------------------------
TITLE_MODEL = setting("TITLE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL = setting("IMAGE_MODEL", "sentence-transformers/clip-ViT-B-32")
ENCODE_THREADS = at_least("ENCODE_THREADS", 4, 1)        # CPU threads for encoding titles
DOWNLOAD_WORKERS = at_least("DOWNLOAD_WORKERS", 8, 1)    # photos fetched at the same time
DOWNLOAD_TIMEOUT = at_least("DOWNLOAD_TIMEOUT", 10, 1, float)   # seconds per photo request
