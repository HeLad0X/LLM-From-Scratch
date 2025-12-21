import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch


# ============================================================
# Time helpers
# ============================================================

def format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_eta(start_time: float, step: int, total_steps: int) -> Tuple[float, float]:
    elapsed = time.time() - start_time
    if step <= 0:
        return elapsed, float("inf")

    time_per_step = elapsed / step
    remaining_steps = max(0, total_steps - step)
    eta = remaining_steps * time_per_step
    return elapsed, eta


# ============================================================
# Logger
# ============================================================

def setup_logger(
    log_dir: str = "./logs",
    name: str = "train",
    file_prefix: str = "train_logs"
) -> logging.Logger:
    """
    Logger that writes to:
    ./logs/train_logs_<timestamp>.log
    and also prints to console.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{file_prefix}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers (important for notebooks)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging started. Log file: {log_path}")
    return logger


# ============================================================
# Checkpointing
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    *,
    model: torch.nn.Module,
    out_dir: str,
    step: int,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    is_final: bool = False,
    prefix: str = "gpt",
) -> str:
    """
    Saves model (and optional optimizer/scheduler) to out_dir.

    NOTE:
    - Embeddings ARE INCLUDED in model.state_dict()
    - Weight tying is preserved
    """
    ensure_dir(out_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "final" if is_final else f"step_{step}"
    file_name = f"{prefix}_{tag}_{timestamp}.pt"
    if is_final:
        out_dir = os.path.join(out_dir, "final")
    else:
        out_dir = os.path.join(out_dir, "checkpoints")

    ensure_dir(out_dir)
    path = os.path.join(out_dir, file_name)

    payload: Dict[str, Any] = {
        "step": step,
        "config": config,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        try:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Loads checkpoint and restores model (+ optimizer/scheduler if provided).
    """
    payload = torch.load(path, map_location=map_location)

    model.load_state_dict(payload["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in payload:
        try:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        except Exception:
            pass

    return payload


# ============================================================
# Training status logger
# ============================================================

def log_train_status(
    logger: logging.Logger,
    *,
    step: int,
    total_steps: int,
    loss: float,
    start_time: float,
    extra: str = "",
) -> None:
    elapsed, eta = compute_eta(start_time, step, total_steps)

    eta_str = "??:??:??" if eta == float("inf") else format_time(eta)

    msg = (
        f"step {step}/{total_steps} | "
        f"loss {loss:.4f} | "
        f"elapsed {format_time(elapsed)} | "
        f"ETA {eta_str}"
    )

    if extra:
        msg += f" | {extra}"

    logger.info(msg)
