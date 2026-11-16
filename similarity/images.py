"""Stage 07 (optional): the main photo of each product.

Standalone: nothing else in the pipeline runs this. Preprocessing, the Docker build, the API
and the evaluation never download a photo. Run from the repo root, in order:

    python -m similarity.images download [--image-dir DIR] [--limit N]
    python -m similarity.images embed [--image-dir DIR]

download  fetches each product's main photo: the first link in the `medium` column, about
          500 px tall (`small` is a 38 x 50 thumbnail, `large` a zoom version of up to 1,500 px).
          Each photo is padded to a white square, so a tall photo is not cropped, shrunk to
          224 x 224 (the size the image model looks at) and saved as <uniq_id>.jpg. A photo
          already saved is skipped, so the download can be stopped and started again. Links
          that fail are listed in failed.csv in the same folder.

embed     turns each saved photo into 512 numbers with CLIP ViT-B/32 (Radford et al., 2021),
          run through sentence-transformers, and writes data/processed/image_embeddings.npz
          (about 61 MB): the numbers, which products have a photo, and the score of a typical
          random pair (see similarity/embeddings.py). Work is saved every 1,000 photos in a
          folder next to the photos, so a stopped run carries on where it stopped.

The photos are kept outside the repo, in IMAGE_DIR (a setting in .env): the repo sits in
OneDrive, and 30,000 photos would sync to the cloud.
"""

from __future__ import annotations

import argparse
import http.client
import io
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from similarity import config
from similarity.embeddings import IMAGE_PATH, save_embeddings
from similarity.preprocessing import load_products

# Settings: see similarity/config.py and .env.example
IMAGE_DIR = config.IMAGE_DIR           # default: Downloads/Amazon Image Download in the home folder
WORKERS = config.DOWNLOAD_WORKERS      # photos fetched at the same time, kept low to be polite to the server
TIMEOUT = config.DOWNLOAD_TIMEOUT      # seconds per request
IMAGE_MODEL = config.IMAGE_MODEL

SIZE = 224                   # pixels per side, what CLIP ViT-B/32 looks at
RETRIES = 2                  # extra tries after a network error, waiting 1 s, then 2 s
STOP_AFTER_REFUSALS = 50     # this many refusals in a row (403, 429, 503) means the server wants us to stop
USER_AGENT = "Mozilla/5.0 (product-similarity research script)"
REFUSED = {403, 429, 503}

CHUNK = 1000                # photos encoded between two saves of the work done
CHUNK_DIR = "clip_chunks"    # inside the photo folder, removed once the final file is written


class Refused(Exception):
    """The server said no (403, 429 or 503), not just a broken link."""


def photo_path(image_dir: Path, uniq_id: str) -> Path:
    return image_dir / f"{uniq_id}.jpg"


def to_square(image: Image.Image, size: int = SIZE) -> Image.Image:
    """Pad to a white square (nothing is cropped), then shrink to size x size."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    return ImageOps.pad(image.convert("RGB"), (size, size), color="white")


def fetch(url: str) -> bytes:
    """The bytes behind a link. Network errors are tried again; a missing page (404) is not."""
    for attempt in range(RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in REFUSED:
                raise
            if attempt == RETRIES:
                raise Refused(f"HTTP {error.code}") from error
        except (OSError, http.client.HTTPException):
            if attempt == RETRIES:
                raise
        time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def save_photo(data: bytes, path: Path) -> None:
    """Square, shrink and save. Written under another name first, so a stopped run leaves no half photo."""
    square = to_square(Image.open(io.BytesIO(data)))
    part = path.with_suffix(".part")
    square.save(part, "JPEG", quality=90)
    part.replace(path)


def download(image_dir: Path = IMAGE_DIR, limit: int | None = None) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    products = load_products()
    linked = products[products["image_url"].notna()]
    todo = linked[[not photo_path(image_dir, u).exists() for u in linked["uniq_id"]]]
    print(f"{len(products):,} products: {len(products) - len(linked):,} without a photo link, "
          f"{len(linked) - len(todo):,} photos already saved, {len(todo):,} to fetch into {image_dir}")
    if limit is not None:
        todo = todo.head(limit)
        print(f"Fetching only the first {len(todo):,} (--limit)")

    stop = threading.Event()

    def work(uniq_id: str, url: str) -> bool:
        if stop.is_set():
            return False
        save_photo(fetch(url), photo_path(image_dir, uniq_id))
        return True

    saved, failed, refusals_in_a_row = 0, [], 0
    start = time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(work, u, url): (u, url) for u, url in zip(todo["uniq_id"], todo["image_url"])}
        for done, future in enumerate(as_completed(futures), start=1):
            uniq_id, url = futures[future]
            try:
                if future.result():
                    saved += 1
                    refusals_in_a_row = 0
            except Refused as error:
                failed.append((uniq_id, url, str(error)))
                refusals_in_a_row += 1
                if refusals_in_a_row >= STOP_AFTER_REFUSALS and not stop.is_set():
                    stop.set()
                    print(f"STOPPED: the server refused {STOP_AFTER_REFUSALS} requests in a row. "
                          "Wait a while before running again.")
            except Exception as error:   # a broken link or a file that is not a picture
                failed.append((uniq_id, url, f"{type(error).__name__}: {error}"))
            if done % 1000 == 0:
                minutes = (time.perf_counter() - start) / 60
                print(f"  {done:,} / {len(todo):,} done, {saved:,} saved, {len(failed):,} failed, {minutes:.1f} min")

    pd.DataFrame(failed, columns=["uniq_id", "url", "error"]).to_csv(image_dir / "failed.csv", index=False)
    minutes = (time.perf_counter() - start) / 60
    total = sum(1 for _ in image_dir.glob("*.jpg"))
    print(f"Saved {saved:,} photos in {minutes:.1f} min; {len(failed):,} failed (listed in failed.csv); "
          f"{total:,} photos in {image_dir}")


def load_photos(image_dir: Path, uniq_ids) -> list[Image.Image]:
    photos = []
    for uniq_id in uniq_ids:
        with Image.open(photo_path(image_dir, uniq_id)) as photo:
            photos.append(photo.convert("RGB"))
    return photos


def embed(image_dir: Path = IMAGE_DIR, batch_size: int = 64) -> None:
    """Encode every saved photo, 1,000 at a time, keeping each finished chunk on disk."""
    from sentence_transformers import SentenceTransformer   # imported here: only this step needs torch

    products = load_products()
    ids = products["uniq_id"].to_numpy()
    has_photo = np.array([photo_path(image_dir, u).exists() for u in ids])
    chunk_dir = image_dir / CHUNK_DIR
    chunk_dir.mkdir(exist_ok=True)

    vectors = np.zeros((len(ids), 512), dtype=np.float32)
    done = np.zeros(len(ids), dtype=bool)
    chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    for chunk in chunks:                                     # work saved by an earlier, stopped run
        with np.load(chunk) as saved:
            vectors[saved["rows"]] = saved["vectors"]
            done[saved["rows"]] = True
    todo = np.flatnonzero(has_photo & ~done)
    print(f"{has_photo.sum():,} products have a photo: {done.sum():,} already encoded, {len(todo):,} to encode")

    model = SentenceTransformer(IMAGE_MODEL, device="cpu")
    start = time.perf_counter()
    for number, first in enumerate(range(0, len(todo), CHUNK), start=len(chunks)):
        rows = todo[first:first + CHUNK]
        vectors[rows] = model.encode(load_photos(image_dir, ids[rows]), batch_size=batch_size,
                                     normalize_embeddings=True, show_progress_bar=False)
        np.savez(chunk_dir / f"chunk_{number:03d}.npz", rows=rows, vectors=vectors[rows])
        finished = first + len(rows)
        minutes = (time.perf_counter() - start) / 60
        print(f"  {finished:,} / {len(todo):,} encoded, {minutes:.1f} min")

    embeddings = save_embeddings(IMAGE_PATH, ids, vectors, has_photo, IMAGE_MODEL)
    for chunk in chunk_dir.glob("chunk_*.npz"):
        chunk.unlink()
    chunk_dir.rmdir()
    print(f"Wrote {IMAGE_PATH}: {has_photo.sum():,} photos x {vectors.shape[1]} numbers, "
          f"{len(ids) - has_photo.sum():,} products without a photo")
    print(f"A typical random pair of photos scores {embeddings.baseline:.2f}; rescaling maps that to 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "embed"])
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR, help="where the photos are kept")
    parser.add_argument("--limit", type=int, default=None, help="download: fetch at most this many (to try it out)")
    args = parser.parse_args()
    if args.step == "download":
        download(args.image_dir, args.limit)
    else:
        embed(args.image_dir)


if __name__ == "__main__":
    main()
