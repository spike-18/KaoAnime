from pathlib import Path
from unittest.mock import MagicMock

import pytest
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

    cb._log_last_to_mlflow(mock_trainer)  # must not raise or call log_artifact


def test_is_subclass_of_model_checkpoint():
    from lightning.pytorch.callbacks import ModelCheckpoint
    assert issubclass(MLflowCheckpointCallback, ModelCheckpoint)
