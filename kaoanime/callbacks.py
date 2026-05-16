from __future__ import annotations

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


class MLflowCheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint that uploads last.ckpt to MLflow after each save.

    Overrides on_train_epoch_end and on_train_end so the checkpoint is
    archived during training. Lightning calls both hooks on graceful
    interrupts (Ctrl+C), ensuring the checkpoint is logged even when
    training is stopped early.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self._log_last_to_mlflow(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        super().on_train_end(trainer, pl_module)
        self._log_last_to_mlflow(trainer)

    def _log_last_to_mlflow(self, trainer) -> None:
        path = self.last_model_path
        if path and isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, path)
