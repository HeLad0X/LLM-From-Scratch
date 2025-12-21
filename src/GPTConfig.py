import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load and return the raw YAML config as a dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ConfigSection:
    """Simple wrapper to allow attribute-style access to dictionaries."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = data or {}
        for key, value in self._data.items():
            setattr(self, key, ConfigSection(value) if isinstance(value, dict) else value)

    def as_dict(self) -> Dict[str, Any]:
        return {
            key: value.as_dict() if isinstance(value, ConfigSection) else value
            for key, value in self._data.items()
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"ConfigSection({self._data})"


class GPTConfig:
    _cfg = load_config(path="config.yaml")

    # =========================
    # Tokenizer configuration
    # =========================
    tokenizer = ConfigSection(_cfg.get("tokenizer"))
    tokenizer_type = getattr(tokenizer, "type", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    specials = getattr(tokenizer, "specials", None)
    min_word_freq = getattr(tokenizer, "min_word_freq", None)
    max_merges = getattr(tokenizer, "max_merges", None)
    token_cache_dir = getattr(tokenizer, "token_cache_dir", None)

    # =========================
    # Data configuration
    # =========================
    data = ConfigSection(_cfg.get("data"))
    data_dir = getattr(data, "data_dir", None)
    dataset_type = getattr(data, "dataset_type", None)
    block_size = getattr(data, "block_size", None)
    stride = getattr(data, "stride", None)
    add_space_between_words = getattr(data, "add_space_between_words", None)
    num_workers = getattr(data, "num_workers", None)
    pin_memory = getattr(data, "pin_memory", None)
    shuffle = getattr(data, "shuffle", None)
    log_dir = getattr(data, "log_dir", None)

    # =========================
    # Training configuration
    # =========================
    training = ConfigSection(_cfg.get("training"))
    batch_size = getattr(training, "batch_size", None)
    grad_accum_steps = getattr(training, "grad_accum_steps", None)
    seed = getattr(training, "seed", None)
    max_steps = getattr(training, "max_steps", None)
    eval_interval = getattr(training, "eval_interval", None)
    log_interval = getattr(training, "log_interval", None)
    grad_clip = getattr(training, "grad_clip", None)

    optimizer = ConfigSection(getattr(training, "_data", {}).get("optimizer"))
    optimizer_name = getattr(optimizer, "name", None)
    lr = getattr(optimizer, "lr", None)
    betas = getattr(optimizer, "betas", None)
    weight_decay = getattr(optimizer, "weight_decay", None)

    scheduler = ConfigSection(getattr(training, "_data", {}).get("scheduler"))
    scheduler_name = getattr(scheduler, "name", None)
    warmup_steps = getattr(scheduler, "warmup_steps", None)
    min_lr = getattr(scheduler, "min_lr", None)

    # =========================
    # Model configuration
    # =========================
    model = ConfigSection(_cfg.get("model"))
    model_vocab_size = getattr(model, "vocab_size", None)
    max_context_length = getattr(model, "max_context_length", None)
    emb_dim = getattr(model, "emb_dim", None)
    n_layers = getattr(model, "n_layers", None)
    n_heads = getattr(model, "n_heads", None)
    dropout = getattr(model, "dropout", None)
    bias = getattr(model, "bias", None)
    norm_eps = getattr(model, "norm_eps", None)
    tie_weights = getattr(model, "tie_weights", None)
    mlp_ratio = getattr(model, "mlp_ratio", None)
    cache_size = getattr(model, "cache_size", None)
    model_cache = getattr(model, "model_cache", None)

    # =========================
    # Runtime / system configuration
    # =========================
    system = ConfigSection(_cfg.get("system"))
    device = getattr(system, "device", None)
    dtype = getattr(system, "dtype", None)
    compile = getattr(system, "compile", None)
    checkpoint_dir = getattr(system, "checkpoint_dir", None)

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return the whole configuration as a plain dictionary."""
        return {
            "tokenizer": cls.tokenizer.as_dict(),
            "data": cls.data.as_dict(),
            "training": cls.training.as_dict(),
            "model": cls.model.as_dict(),
            "system": cls.system.as_dict(),
        }
