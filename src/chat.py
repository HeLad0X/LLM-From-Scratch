import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from GPTConfig import GPTConfig
from GPTModel import GPTModel
from TrainUtils import load_checkpoint
from preprocessing.BPETokenizer import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = BPETokenizer.load(GPTConfig.token_cache_dir)
model = GPTModel(GPTConfig).to(device)

ckpt_path = "cache_data\\model_cache\\final\\gpt_final_20251221_080446.pt"  # update to your file
load_checkpoint(ckpt_path, model=model, map_location=device)

model.eval()

text = "Once upon a time to the world, and the day on the next years on the first world of one of the best of the time. But the first time. It's have"
encode = tok.encode(text)

print(encode)
decode = tok.decode(encode)
print(decode)
print(text == decode)


@torch.no_grad()
def sample_next_id(ids, temperature=1.0, top_k=50):
    ids = ids[-GPTConfig.max_context_length:]  # keep within pos_emb limit
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    logits = model(x)[0, -1] / max(temperature, 1e-8)

    if top_k is not None:
        v, ix = torch.topk(logits, top_k)
        probs = torch.softmax(v, dim=-1)
        next_id = ix[torch.multinomial(probs, 1)].item()
    else:
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
    return next_id

def generate(prompt, max_new=50, temperature=0.9, top_k=50):
    ids = tok.encode(prompt)
    for _ in range(max_new):
        ids.append(sample_next_id(ids, temperature=temperature, top_k=top_k))
        eos_id = tok.stoi.get("<EOS>")
        if eos_id is not None and ids[-1] == eos_id:
            break
    return tok.decode(ids)

# print(generate("Once upon a time", max_new=60, temperature=0.7, top_k=20))



def format_chat(history, user_msg):
    parts = []
    for u, a in history:
        parts.append(f"User: {u}\nAssistant: {a}\n")
    parts.append(f"User: {user_msg}\nAssistant:")
    return "".join(parts)

# history = []
# user_msg = "Explain gradient clipping."
# prompt = format_chat(history, user_msg)
# reply = generate(prompt, max_new=120)
# print(reply)
