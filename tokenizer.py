import re
from config import normalize_text

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text



class SimpleTokenizerV5:
    def __init__(self, vocab, unk_token='<|unk|>', nl_token='<|NL|>'):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.unk_token = unk_token
        self.nl_token = nl_token

    def encode(self, text):
        # assume you've added normalize_text in both vocab build + here
        parts = re.split(r'([,.:;?_!"()\']|--|\n|\s)', text)
        toks = []
        for p in parts:
            if not p:
                continue
            if p == "\n":
                toks.append(self.nl_token)
            elif p.isspace():
                continue
            else:
                toks.append(p)
        unk_id = self.str_to_int[self.unk_token]
        return [self.str_to_int.get(t, unk_id) for t in toks]

    def decode(self, ids):
        T = [self.int_to_str[i] for i in ids]
        nl = self.nl_token
        attach_right = {",", ".", ":", ";", "?", "!", ")", "]", "}"}
        attach_left  = {"(", "[", "{"}

        out = []
        last_char = ""   # last emitted character

        i = 0
        n = len(T)
        while i < n:
            t = T[i]

            # hard newline
            if t == nl:
                # trim trailing space before newline
                if out and out[-1].endswith(" "):
                    out[-1] = out[-1].rstrip()
                out.append("\n")
                last_char = "\n"
                i += 1
                continue

            # double dash → normalize as " -- "
            if t == "--":
                if out and out[-1].endswith(" "):
                    out[-1] = out[-1].rstrip()
                out.append(" -- ")
                last_char = " "
                i += 1
                continue

            # quotes: opening vs closing based on previous char
            if t == '"':
                # opening if at start/after whitespace/opening bracket/dash/newline
                if (not last_char) or last_char.isspace() or last_char in "([{—-":
                    # remove any stray space before opening quote
                    if out and out[-1].endswith(" "):
                        out[-1] = out[-1].rstrip()
                    out.append('"')
                    last_char = '"'
                else:
                    # closing quote
                    if out and out[-1].endswith(" "):
                        out[-1] = out[-1].rstrip()
                    out.append('"')
                    last_char = '"'
                    # add a following space if next token is a word or opening bracket/quote
                    if i + 1 < n and T[i+1] not in {nl} | attach_right:
                        out.append(" ")
                        last_char = " "
                i += 1
                continue

            # apostrophe: glue to neighbors (we’ll also regex-fix later)
            if t == "'":
                if out and out[-1].endswith(" "):
                    out[-1] = out[-1].rstrip()
                out.append("'")
                last_char = "'"
                i += 1
                continue

            # punctuation that attaches to the left (no space before)
            if t in attach_right:
                if out and out[-1].endswith(" "):
                    out[-1] = out[-1].rstrip()
                out.append(t)
                out.append(" ")  # space after , . : ; ? !  ) ] }
                last_char = " "
                i += 1
                continue

            # opening brackets attach to the right (no space after)
            if t in attach_left:
                if out and out[-1].endswith(" "):
                    out[-1] = out[-1].rstrip()
                out.append(t)
                last_char = t
                i += 1
                continue

            # default: word-like token
            if not out:
                out.append(t)
            else:
                # no space if last was an opening bracket or opening quote or newline
                if last_char in {'"', "(", "[", "{", "\n"}:
                    out.append(t)
                else:
                    if last_char != " ":
                        out.append(" ")
                    out.append(t)
            last_char = t[-1] if t else last_char
            i += 1

        text = "".join(out)

        # --- targeted cleanups ---

        # remove any space before right-attaching punctuation
        text = re.sub(r"\s+([,.:;?!\)\]\}])", r"\1", text)

        # ensure space after punctuation if followed by a word/quote
        text = re.sub(r"([,.:;?!])(?=(?:[A-Za-z0-9\"(\[{]))", r"\1 ", text)

        # no space after opening brackets/quotes
        text = re.sub(r'([(\[\{])\s+', r"\1", text)
        text = re.sub(r'"\s+', r'"', text)

        # no space before closing quote
        text = re.sub(r'\s+"', r'"', text)

        # apostrophes inside words
        text = re.sub(r"(\w)\s+'\s+(\w)", r"\1'\2", text)
        text = re.sub(r"(\w)\s+'\s+s\b", r"\1's", text)

        # normalize spaces around --
        text = re.sub(r"\s*--\s*", r" -- ", text)

        # collapse multiple spaces, but keep newlines tidy
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)

        return text.strip()
