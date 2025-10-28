import tiktoken
from utility import DATA_PATH

tokenizer = tiktoken.get_encoding("gpt2")

raw_text = None

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)

encoded_sample = encoded_text[50:]
context_size = 4

x = encoded_sample[:context_size]
y = encoded_sample[1:context_size+1]

print(x)
print(y)