import time
from contextlib import nullcontext
import torch
import torch.nn.functional as F

from GPTConfig import GPTConfig
from GPTModel import GPTModel
from preprocessing.GetDataset import get_train_val_loaders
from preprocessing.TokenizerFactory import load_tokenizer
from TrainUtils import setup_logger, log_train_status, save_checkpoint


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


def main():
    logger = setup_logger(log_dir=getattr(GPTConfig, "log_dir", "./logs"), name="train")

    device = getattr(GPTConfig, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    use_cuda = str(device).startswith("cuda")

    if use_cuda:
        # Speed-oriented, but also helps keep step time reasonable when using memory-saving features.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype_name = str(getattr(GPTConfig, "dtype", "float32") or "float32").lower()
    amp_dtype = None
    if use_cuda and dtype_name in {"float16", "fp16", "half"}:
        amp_dtype = torch.float16
    elif use_cuda and dtype_name in {"bfloat16", "bf16"}:
        amp_dtype = torch.bfloat16

    amp_enabled = amp_dtype is not None
    scaler = torch.amp.GradScaler(device="cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    logger.info(f"AMP: {'on' if amp_enabled else 'off'} | dtype: {dtype_name}")

    train_loader, val_loader = get_train_val_loaders()
    tok = load_tokenizer()
    if len(tok) != int(GPTConfig.vocab_size):
        raise ValueError(f"Tokenizer vocab {len(tok)} != config tokenizer.vocab_size {GPTConfig.vocab_size}")
    if int(GPTConfig.model_vocab_size) != int(GPTConfig.vocab_size):
        raise ValueError(
            f"config model.vocab_size ({GPTConfig.model_vocab_size}) "
            f"!= tokenizer.vocab_size ({GPTConfig.vocab_size})"
        )

    model = GPTModel(GPTConfig).to(device)

    if bool(getattr(GPTConfig, "tie_weights", False)):
        model.out_head.weight = model.tok_emb.weight
        logger.info("Weight tying enabled")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GPTConfig.lr,
        betas=tuple(GPTConfig.betas),
        weight_decay=GPTConfig.weight_decay
    )

    grad_accum = int(getattr(GPTConfig, "grad_accum_steps", 1))
    grad_clip = float(getattr(GPTConfig, "grad_clip", 1.0))
    log_interval = int(getattr(GPTConfig, "log_interval", 50))
    eval_interval = int(getattr(GPTConfig, "eval_interval", 500))
    max_steps = int(getattr(GPTConfig, "max_steps", 1000))

    warmup_steps = int(getattr(GPTConfig, "warmup_steps", 0) or 0)
    if warmup_steps >= max_steps:
        warmup_steps = max(max_steps - 1, 0)

    schedulers = []
    milestones = []
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        schedulers.append(warmup)

    cosine_steps = max_steps - warmup_steps
    if cosine_steps > 0:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_steps,
            eta_min=GPTConfig.min_lr,
        )
        schedulers.append(cosine)
        if warmup_steps > 0:
            milestones = [warmup_steps]

    if len(schedulers) == 2:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer, schedulers=schedulers, milestones=milestones
        )
    elif len(schedulers) == 1:
        scheduler = schedulers[0]
    else:
        scheduler = None

    ckpt_dir = getattr(GPTConfig, "model_cache", None) or getattr(GPTConfig, "checkpoint_dir", "./checkpoints")
    config_dict = GPTConfig.as_dict()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    step = 0
    micro_step = 0

    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled else nullcontext()
        )
        with amp_ctx:
            logits = model(x)
            loss = compute_loss(logits, y) / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % log_interval == 0:
                log_train_status(
                    logger,
                    step=step,
                    total_steps=max_steps,
                    loss=loss.item() * grad_accum,
                    start_time=start_time,
                )

            if val_loader is not None and step > 0 and step % eval_interval == 0:
                val_loss = evaluate(
                    model,
                    val_loader,
                    device=device,
                    max_batches=50,
                    amp_dtype=amp_dtype,
                )
                logger.info(f"eval @ step {step} | val_loss {val_loss:.4f}")

                ckpt = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    out_dir=ckpt_dir,
                    step=step,
                    config=config_dict,
                    is_final=False,
                    prefix="gpt"
                )
                logger.info(f"Saved checkpoint: {ckpt}")

            step += 1

    final_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        out_dir=ckpt_dir,
        step=step,
        config=config_dict,
        is_final=True,
        prefix="gpt"
    )
    logger.info(f"Training complete. Final saved: {final_path}")


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    max_batches: int = 50,
    amp_dtype=None,
) -> float:
    model.eval()
    losses = []
    amp_enabled = (amp_dtype is not None) and str(device).startswith("cuda")
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled else nullcontext()
        )
        with amp_ctx:
            logits = model(x)
            loss = compute_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    main()
