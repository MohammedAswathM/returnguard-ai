from pathlib import Path
from urllib.request import urlopen

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_DOI = "10.24432/C5CG6D"
UCI_LICENSE = "CC BY 4.0"


def download_uci(destination: Path) -> Path:
    """Download the official UCI archive without interpreting cancellations as labels."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(UCI_URL, timeout=120) as response, partial.open("wb") as handle:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination
