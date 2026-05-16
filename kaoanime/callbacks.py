from __future__ import annotations

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


class MLflowCheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint that uploads last.ckpt to MLflow after each save.

    on_train_epoch_end uploads after each completed epoch, so the latest
    checkpoint is always in MLflow. on_train_end catches the case where
    Lightning deferred the final save to train end (e.g. no validation loop).
    Note: mid-epoch interrupts (Ctrl+C) call on_exception, not on_train_end,
    and are not covered by this callback.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self._log_last_to_mlflow(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        path_before = self.last_model_path
        super().on_train_end(trainer, pl_module)
        if self.last_model_path and self.last_model_path != path_before:
            self._log_last_to_mlflow(trainer)

    def _log_last_to_mlflow(self, trainer) -> None:
        path = self.last_model_path
        if path and isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, path)
