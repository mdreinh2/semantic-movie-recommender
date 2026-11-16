"""Download the exact raw inputs used by this handoff."""

import hashlib
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMDB_COMMIT = "287abcb9999f332e13d93b17c9b0352b96875941"
HASHES = {
    "movies_metadata.csv": "37ec322786b136a24604564e55160be75f8937546f11cb1a990076971fa895fd",
    "keywords.csv": "428ef94a80255986ad71fd347855702e772ee776841f1e8923a618aa2d87a668",
    "credits.csv": "b968359c8e77aebcb7329608378b3b11cc1834726e157aae934bc89f291458ec",
    "ratings.csv": "784e85369bbdbbb32e54cbec653a79d087ae1525ffbc5b875de952b72316c00e",
    "links.csv": "5c8826f9523d6d004046ef579d42b655c58f7c5d55a868913180cf11e3e31c93",
}


def digest(path, algorithm="sha256"):
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def download(url, target, expected, algorithm="sha256"):
    if target.exists() and digest(target, algorithm) == expected:
        print(f"Verified existing {target.name}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        pending = Path(temporary) / target.name
        for attempt in range(3):
            print(f"Downloading {target.name} (attempt {attempt + 1})", flush=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "CSE573-Member1/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, pending.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if digest(pending, algorithm) != expected:
                    raise ValueError(f"Checksum mismatch for {target.name}.")
                pending.replace(target)
                return
            except (OSError, urllib.error.URLError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))


def main():
    raw = ROOT / "data" / "raw"
    ml = raw / "ml-20m"
    ml.mkdir(parents=True, exist_ok=True)
    if not all((ml / name).exists() and digest(ml / name) == HASHES[name]
               for name in ("ratings.csv", "links.csv")):
        archive = raw / "ml-20m.zip"
        download(
            "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
            archive, "cd245b17a1ae2cc31bb14903e1204af3", "md5",
        )
        with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(dir=ml) as temporary:
            for name in ("ratings.csv", "links.csv"):
                pending = Path(temporary) / name
                with source.open(f"ml-20m/{name}") as data, pending.open("wb") as target:
                    shutil.copyfileobj(data, target)
                if digest(pending) != HASHES[name]:
                    raise ValueError(f"Unexpected MovieLens contents for {name}.")
                pending.replace(ml / name)
    for name in ("movies_metadata.csv", "keywords.csv", "credits.csv"):
        download(
            f"https://huggingface.co/datasets/gaygaaa/TheMoviesDataset/resolve/{TMDB_COMMIT}/{name}",
            raw / "tmdb" / name, HASHES[name],
        )
    print("All five source files verified.", flush=True)


if __name__ == "__main__":
    main()
