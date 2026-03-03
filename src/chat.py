import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from GPTConfig import GPTConfig
from GPTModel import GPTModel
from TrainUtils import load_checkpoint
from preprocessing.TokenizerFactory import load_tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"


def _latest_checkpoint() -> str:
    base_dir = Path(getattr(GPTConfig, "model_cache", None) or "cache_data/model_cache")
    final_dir = base_dir / "final"
    ckpt_dir = base_dir / "checkpoints"

    finals = sorted(final_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if finals:
        return str(finals[0])

    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return str(ckpts[0])

    raise FileNotFoundError(f"No checkpoint found under {base_dir}")


tok = load_tokenizer()
if len(tok) != int(GPTConfig.vocab_size):
    raise ValueError(f"Tokenizer vocab {len(tok)} != config tokenizer.vocab_size {GPTConfig.vocab_size}")
if int(GPTConfig.model_vocab_size) != int(GPTConfig.vocab_size):
    raise ValueError(
        f"config model.vocab_size ({GPTConfig.model_vocab_size}) "
        f"!= tokenizer.vocab_size ({GPTConfig.vocab_size})"
    )
model = GPTModel(GPTConfig).to(device)
ckpt_path = _latest_checkpoint()
payload = load_checkpoint(ckpt_path, model=model, map_location=device)
model.eval()

if model.tok_emb.num_embeddings != len(tok):
    raise ValueError(
        f"Checkpoint/model vocab mismatch after load: model embeddings={model.tok_emb.num_embeddings}, "
        f"tokenizer vocab={len(tok)}. Check tokenizer/config/checkpoint directory."
    )

EOS_ID = tok.stoi.get("<EOS>")
PAD_ID = tok.stoi.get("<PAD>")
BOS_ID = tok.stoi.get("<BOS>")
UNK_ID = tok.stoi.get("<UNK>")


def _apply_repetition_penalty(logits, ids, penalty=1.10, window=128):
    if penalty is None or penalty <= 1.0 or not ids:
        return logits
    recent = ids[-window:]
    for tid in set(recent):
        if logits[tid] < 0:
            logits[tid] *= penalty
        else:
            logits[tid] /= penalty
    return logits


def _apply_forbidden_token_bias(logits):
    # Prevent non-text control-ish tokens from being sampled.
    for tid in (PAD_ID, BOS_ID):
        if tid is not None:
            logits[tid] = -float("inf")
    # Strongly discourage UNK; allow it only if the model is desperate.
    if UNK_ID is not None:
        logits[UNK_ID] -= 5.0
    return logits


def _apply_no_repeat_ngram(logits, ids, ngram_size=3):
    if ngram_size is None or ngram_size <= 1 or len(ids) < ngram_size - 1:
        return logits

    prefix = tuple(ids[-(ngram_size - 1):])
    banned = set()
    for i in range(len(ids) - ngram_size + 1):
        if tuple(ids[i : i + ngram_size - 1]) == prefix:
            banned.add(ids[i + ngram_size - 1])

    for tid in banned:
        logits[tid] = -float("inf")
    return logits


def _sample_from_logits(logits, temperature=0.65, top_k=20, top_p=0.90, greedy=False):
    if greedy or temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(float(temperature), 1e-8)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.numel())
        v, ix = torch.topk(logits, k)
        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits[ix] = v
        logits = filtered_logits

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)

        remove = cum > float(top_p)
        remove[0] = False
        sorted_logits[remove] = -float("inf")

        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits[sorted_idx] = sorted_logits
        logits = filtered_logits

    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def sample_next_id(
    ids,
    *,
    temperature=0.65,
    top_k=20,
    top_p=0.90,
    greedy=False,
    repetition_penalty=1.10,
    repetition_window=128,
    no_repeat_ngram_size=3,
    min_new_tokens=0,
    generated_so_far=0,
):
    ids = ids[-GPTConfig.max_context_length:]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    logits = model(x)[0, -1].float()

    logits = _apply_forbidden_token_bias(logits)
    logits = _apply_repetition_penalty(logits, ids, penalty=repetition_penalty, window=repetition_window)
    logits = _apply_no_repeat_ngram(logits, ids, ngram_size=no_repeat_ngram_size)

    if EOS_ID is not None and generated_so_far < min_new_tokens:
        logits[EOS_ID] = -float("inf")

    return _sample_from_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
    )


def generate(
    prompt,
    *,
    max_new=80,
    temperature=0.40,
    top_k=None,
    top_p=0.85,
    greedy=False,
    repetition_penalty=1.20,
    repetition_window=128,
    no_repeat_ngram_size=4,
    min_new_tokens=0,
):
    ids = tok.encode(prompt)
    prompt_len = len(ids)

    if prompt_len == 0 and BOS_ID is not None:
        ids = [BOS_ID]
        prompt_len = 1

    for step in range(max_new):
        next_id = sample_next_id(
            ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            generated_so_far=step,
        )
        ids.append(next_id)
        if EOS_ID is not None and next_id == EOS_ID:
            break

    return tok.decode(ids), tok.decode(ids[prompt_len:])


def format_chat(history, user_msg):
    parts = []
    for u, a in history:
        parts.append(f"User: {u}\nAssistant: {a}\n")
    parts.append(f"User: {user_msg}\nAssistant:")
    return "".join(parts)


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    if isinstance(payload, dict) and "step" in payload:
        print(f"Checkpoint step: {payload['step']}")

    test_prompts = [
        ("news-like", "The government said on Tuesday that"),
        ("wiki-like", "Machine learning is a field of artificial intelligence that"),
        ("story-like", "Once upon a time"),
    ]

    for label, prompt in test_prompts:
        full, continuation = generate(
            prompt,
            max_new=80,
            temperature=0.45,
            top_k=None,
            top_p=0.85,
            greedy=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )
        print(f"\n[{label}] prompt: {prompt}")
        print(f"[{label}] continuation: {continuation}")

    # For raw quality inspection, try greedy decoding:
    # full, continuation = generate("The company announced that", greedy=True, max_new=60, temperature=0.0)
    # print(continuation)
