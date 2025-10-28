import os, json, time, re
from pathlib import Path
from urllib.parse import urlparse
import requests

API_BASE = "https://gutendex.com/books"
OUT_DIR = Path("data/raw/gutenberg_top100")
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = OUT_DIR / "metadata.json"

PREFERRED_MIME_ORDER = [
    "text/plain; charset=utf-8",
    "text/plain; charset=us-ascii",
    "text/plain",
]

def pick_text_url(formats: dict):
    # Prefer text/plain UTF-8, then other text/plain variants; else None
    for key in PREFERRED_MIME_ORDER:
        if key in formats:
            return formats[key]
    # Some entries only have zipped text; you can optionally handle zips later
    for k, v in formats.items():
        if k.startswith("text/plain"):
            return v
    return None

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/*?\"<>|:]", "_", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:120]

def download(url: str, dest: Path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with requests.get(url, timeout=30, allow_redirects=True) as r:
                r.raise_for_status()
                dest.write_bytes(r.content)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"FAILED: {url} -> {e}")
                return False
            time.sleep(1.5 * (attempt + 1))

def fetch_top100(languages="en"):
    books = []
    seen_ids = set()
    url = f"{API_BASE}?sort=popular&languages={languages}"
    while url and len(books) < 100:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        for b in data["results"]:
            if b["id"] in seen_ids:
                continue
            text_url = pick_text_url(b.get("formats", {}))
            if not text_url:
                continue
            books.append({**b, "_text_url": text_url})
            seen_ids.add(b["id"])
            if len(books) >= 100:
                break
        url = data.get("next")
        # be polite
        time.sleep(0.3)
    return books[:100]

def main():
    books = fetch_top100(languages="en")
    meta = []
    for b in books:
        gid = b["id"]
        title = b["title"]
        text_url = b["_text_url"]
        # Construct a neat filename
        author = b["authors"][0]["name"] if b["authors"] else "Unknown"
        fname = sanitize_filename(f"{gid} - {title} - {author}.txt")
        dest = OUT_DIR / fname

        ok = download(text_url, dest)
        if not ok:
            continue

        meta.append({
            "gutenberg_id": gid,
            "title": title,
            "authors": b["authors"],
            "languages": b["languages"],
            "download_count": b.get("download_count"),
            "text_url": text_url,
            "saved_as": str(dest),
        })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(meta)} books to {OUT_DIR}")
    print(f"Metadata: {META_PATH}")

if __name__ == "__main__":
    main()
