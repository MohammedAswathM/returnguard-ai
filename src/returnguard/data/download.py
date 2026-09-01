from pathlib import Path
from urllib.request import urlretrieve

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_DOI = "10.24432/C5CG6D"
UCI_LICENSE = "CC BY 4.0"


def download_uci(destination: Path) -> Path:
    """Download the official UCI archive without interpreting cancellations as labels."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(UCI_URL, destination)  # noqa: S310 - locked, documented HTTPS source
    return destination

