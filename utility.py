import os
import unicodedata
import re

cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, 'data/the_verdict.txt')

# 1) same normalization used in BOTH vocab-building and encode()
def normalize_text(t: str) -> str:
    # Unicode normalize
    t = unicodedata.normalize("NFKC", t)

    # Map curly quotes/apostrophes to ASCII
    t = t.replace("‘", "'").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"')

    # Map dashes and ellipsis
    t = t.replace("—", "--").replace("–", "-")   # choose one convention
    t = t.replace("…", "...")

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Optional: collapse weird spaces to regular space
    t = re.sub(r"\s+", lambda m: "\n" if "\n" in m.group(0) else " ", t)

    return t