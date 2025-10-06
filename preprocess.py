import re
from config import DATA_PATH, normalize_text
from tokenizer import SimpleTokenizerV5

def create_vocab():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_text = normalize_text(raw_text)

    # include \n explicitly so lines become tokens (we'll map to <|NL|>)
    parts = re.split(r'([,.:;?_!"()\']|--|\n|\s)', raw_text)
    tokens = [p.strip() for p in parts if p and p.strip()]

    word_vocab = sorted(set(tokens))

    # specials (make sure they exist even if unseen)
    word_vocab.extend(["<|endoftext|>", "<|unk|>", "<|NL|>"])

    # dedupe while preserving last occurrence of specials
    word_vocab = list(dict.fromkeys(word_vocab))

    vocab = {tok: i for i, tok in enumerate(word_vocab)}
    return vocab

def preprocess_data():
    vocab = create_vocab()
    tokenizer = SimpleTokenizerV5(vocab)

    sample_text = """
    Hello.
    Grafton Gallery show, stopped me before Gisburn's "Moon-dancers"

to say, with tears in her eyes: "We shall not look upon

its like again"?
    """

    ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(ids)
    print(decoded_text)

preprocess_data()