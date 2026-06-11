class NullLogger:
    def log(self, metrics: dict, step: int | None = None):
        pass

    def finish(self):
        pass


class ConsoleLogger:
    def log(self, metrics: dict, step: int | None = None):
        prefix = f"[step {step}] " if step is not None else ""
        parts = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        print(f"{prefix}{parts}")

    def finish(self):
        pass


class WandbLogger:
    def __init__(self, project, name=None, config=None, enabled=True):
        self._run = None
        if not enabled:
            return
        import wandb

        self._run = wandb.init(project=project, name=name, config=config)

    def log(self, metrics: dict, step: int | None = None):
        if self._run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def finish(self):
        if self._run is not None:
            import wandb

            wandb.finish()
            self._run = None
