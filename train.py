# train.py
import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.config import Config, register_configs
from kaoanime.model import KaoAnimeModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader

torch.set_float32_matmul_precision("medium")
register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    dataset = UnpairedImageDataset(
        cfg.data.root_a,
        cfg.data.root_b,
        cfg.data.image_size,
        extra_roots_a=list(cfg.data.extra_roots_a),
        extra_roots_b=list(cfg.data.extra_roots_b),
        align_a=cfg.data.align_a,
    )

    train_dl = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = KaoAnimeModel(cfg)

    logger = MLFlowLogger(
        experiment_name="kaoanime",
        tracking_uri=cfg.train.mlflow_tracking_uri,
    )

    ckpt_callback = ModelCheckpoint(filename="epoch{epoch:03d}", save_last=True, save_top_k=0)

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=logger,
        callbacks=[ckpt_callback],
    )
    trainer.fit(model, train_dl)

    if ckpt_callback.last_model_path:
        logger.experiment.log_artifact(logger.run_id, ckpt_callback.last_model_path)


if __name__ == "__main__":
    main()
