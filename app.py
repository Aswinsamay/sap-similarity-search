"""Stage 04: the web API around find_similar_products.

Run from the repo root, after `python -m similarity.preprocessing`:

    uvicorn app:app --port 8000

then open http://localhost:8000/docs to try it in the browser.

GET /find_similar_products?product_id=<uniq_id>&num_similar=10
    200  JSON list of uniq_ids, most similar first
    404  product_id is not in the data
    422  product_id missing or empty, or num_similar not a whole number from 1 to 100
    500  anything unexpected (the details go to the log, not to the caller)

GET /health
    200  {"status": "ok", "products": 30000}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from similarity.search import find_similar_products, load_features

MAX_SIMILAR = 100   # the most a caller may ask for, and how many are cached per product
CACHE_SIZE = 1024   # how many products' results are kept in memory

# uvicorn's own logger, so these messages appear in the server output.
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the features once at startup. uvicorn waits for this before accepting requests."""
    features = load_features()
    logger.info("loaded %d products", len(features.ids))
    yield


app = FastAPI(title="Product similarity", lifespan=lifespan)


@lru_cache(maxsize=CACHE_SIZE)
def top_similar(product_id: str) -> tuple[str, ...]:
    """The MAX_SIMILAR best matches for a product, searched once and then kept.

    Any num_similar up to MAX_SIMILAR is a slice of this. A tuple, so no caller
    can change the kept answer.
    """
    return tuple(find_similar_products(product_id, MAX_SIMILAR))


@app.get("/find_similar_products")
def get_similar_products(
    product_id: Annotated[str, Query(min_length=1, description="uniq_id of the product to match")],
    num_similar: Annotated[int, Query(ge=1, le=MAX_SIMILAR, description="how many products to return")] = 10,
) -> list[str]:
    # Checked here, before the cache, so unknown ids are never stored in it.
    if product_id not in load_features().row_of:
        raise HTTPException(status_code=404, detail=f"product_id not found: {product_id}")
    try:
        return list(top_similar(product_id)[:num_similar])
    except Exception:
        logger.exception("search failed for product_id %s", product_id)
        raise HTTPException(status_code=500, detail="internal error while searching")


@app.get("/health")
def health() -> dict:
    """For Docker and Kubernetes: answers only once the features are loaded."""
    return {"status": "ok", "products": len(load_features().ids)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
