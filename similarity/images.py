"""Stage 07 (optional): the main photo of each product.

Standalone: nothing else in the pipeline runs this. Preprocessing, the Docker build, the API
and the evaluation never download a photo. Run from the repo root:

    python -m similarity.images download [--image-dir DIR] [--limit N]

download  fetches each product's main photo: the first link in the `medium` column, about
          500 px tall (`small` is a 38 x 50 thumbnail, `large` a zoom version of up to 1,500 px).
          Each photo is padded to a white square, so a tall photo is not cropped, shrunk to
          224 x 224 (the size the image model looks at) and saved as <uniq_id>.jpg. A photo
          already saved is skipped, so the download can be stopped and started again. Links
          that fail are listed in failed.csv in the same folder.

The photos are kept outside the repo, in IMAGE_DIR: the repo sits in OneDrive, and 30,000
photos would sync to the cloud.
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

import pandas as pd
from PIL import Image, ImageOps

from similarity.preprocessing import load_products

IMAGE_DIR = Path(r"C:\Users\aswin\Downloads\Amazon Image Download")
SIZE = 224                   # pixels per side, what CLIP ViT-B/32 looks at
WORKERS = 8                  # photos fetched at the same time, kept low to be polite to the server
TIMEOUT = 10                 # seconds per request
RETRIES = 2                  # extra tries after a network error, waiting 1 s, then 2 s
STOP_AFTER_REFUSALS = 50     # this many refusals in a row (403, 429, 503) means the server wants us to stop
USER_AGENT = "Mozilla/5.0 (product-similarity research script)"
REFUSED = {403, 429, 503}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download"])
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR, help="where the photos are kept")
    parser.add_argument("--limit", type=int, default=None, help="fetch at most this many (to try it out)")
    args = parser.parse_args()
    if args.step == "download":
        download(args.image_dir, args.limit)


if __name__ == "__main__":
    main()
