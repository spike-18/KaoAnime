from __future__ import annotations

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


class MLflowCheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint that saves last.ckpt at every epoch end and uploads it to MLflow.

    Lightning 2.6.1 gates _save_last_checkpoint inside on_train_epoch_end on a
    top-k checkpoint having been saved in the same global step.  With
    save_top_k=0 that gate never opens, so last.ckpt is never written and
    last_model_path stays empty.  This subclass forces the save directly.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self.save_last:
            # Lightning 2.6.1 skips _save_last_checkpoint when save_top_k=0
            # (gated on _last_global_step_saved == global_step, which is only
            # updated by _save_topk_checkpoint).  Force it here.
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
        self._log_last_to_mlflow(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        super().on_train_end(trainer, pl_module)
        self._log_last_to_mlflow(trainer)

    def _log_last_to_mlflow(self, trainer) -> None:
        path = self.last_model_path
        if path and isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, path)
