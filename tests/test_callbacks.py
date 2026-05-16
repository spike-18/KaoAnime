from unittest.mock import MagicMock, patch

from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.callbacks import MLflowCheckpointCallback


def test_logs_artifact_when_last_path_set(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cb.last_model_path = str(ckpt)

    mock_logger = MagicMock(spec=MLFlowLogger)
    mock_logger.run_id = "run-abc"
    mock_trainer = MagicMock()
    mock_trainer.logger = mock_logger

    cb._log_last_to_mlflow(mock_trainer)

    mock_logger.experiment.log_artifact.assert_called_once_with("run-abc", str(ckpt))


def test_does_not_log_when_no_checkpoint(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    cb.last_model_path = ""

    mock_logger = MagicMock(spec=MLFlowLogger)
    mock_trainer = MagicMock()
    mock_trainer.logger = mock_logger

    cb._log_last_to_mlflow(mock_trainer)

    mock_logger.experiment.log_artifact.assert_not_called()


def test_does_not_log_when_logger_not_mlflow(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cb.last_model_path = str(ckpt)

    mock_trainer = MagicMock()
    mock_trainer.logger = MagicMock()  # generic logger, not MLFlowLogger

    cb._log_last_to_mlflow(mock_trainer)
    mock_trainer.logger.experiment.log_artifact.assert_not_called()


def test_is_subclass_of_model_checkpoint():
    from lightning.pytorch.callbacks import ModelCheckpoint
    assert issubclass(MLflowCheckpointCallback, ModelCheckpoint)


def test_on_train_epoch_end_calls_super_and_logs(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cb.last_model_path = str(ckpt)

    mock_logger = MagicMock(spec=MLFlowLogger)
    mock_logger.run_id = "run-epoch"
    mock_trainer = MagicMock()
    mock_trainer.logger = mock_logger

    with patch("lightning.pytorch.callbacks.ModelCheckpoint.on_train_epoch_end") as mock_super:
        cb.on_train_epoch_end(mock_trainer, MagicMock())
        mock_super.assert_called_once()

    mock_logger.experiment.log_artifact.assert_called_once_with("run-epoch", str(ckpt))


def test_on_train_end_calls_super_and_logs(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt_new = tmp_path / "last.ckpt"
    ckpt_new.touch()

    mock_logger = MagicMock(spec=MLFlowLogger)
    mock_logger.run_id = "run-end"
    mock_trainer = MagicMock()
    mock_trainer.logger = mock_logger

    def fake_super_on_train_end(trainer, pl_module):
        cb.last_model_path = str(ckpt_new)

    with patch(
        "lightning.pytorch.callbacks.ModelCheckpoint.on_train_end",
        side_effect=fake_super_on_train_end,
    ) as mock_super:
        cb.on_train_end(mock_trainer, MagicMock())
        mock_super.assert_called_once()

    mock_logger.experiment.log_artifact.assert_called_once_with("run-end", str(ckpt_new))


def test_does_not_log_when_logger_is_none(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cb.last_model_path = str(ckpt)

    mock_trainer = MagicMock()
    mock_trainer.logger = None

    cb._log_last_to_mlflow(mock_trainer)  # must not raise
