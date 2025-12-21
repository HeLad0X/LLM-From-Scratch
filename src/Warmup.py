# Warmup.py

class LinearWarmup:
    """
    Linearly increases learning rate from 0 → base_lr
    over `warmup_steps` optimizer steps.
    """

    def __init__(self, optimizer, warmup_steps: int, base_lr: float):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")

        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.base_lr = float(base_lr)
        self._step = 0

        # start at 0 LR
        self._set_lr(0.0)

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self) -> float:
        """
        Returns LR for the *current* step.
        """
        progress = min(self._step / self.warmup_steps, 1.0)
        return self.base_lr * progress

    def step(self) -> float:
        """
        Call once per optimizer step.
        """
        self._step += 1
        lr = self.get_lr()
        self._set_lr(lr)
        return lr

    def is_finished(self) -> bool:
        return self._step >= self.warmup_steps

    # ---------- checkpointing ----------
    def state_dict(self):
        return {
            "step": self._step,
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state):
        self._step = state["step"]
        self.warmup_steps = state["warmup_steps"]
        self.base_lr = state["base_lr"]
        self._set_lr(self.get_lr())
