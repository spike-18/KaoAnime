# train.py
import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.config import Config, register_configs
from kaoanime.model_cyclegan import KaoAnimeModel
from kaoanime.model_not import NOTModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader

torch.set_float32_matmul_precision("medium")
register_configs()


def _make_model(cfg: Config) -> pl.LightningModule:
    if cfg.model_type == "cyclegan":
        return KaoAnimeModel(cfg)
    if cfg.model_type == "not":
        return NOTModel(cfg)
    raise ValueError(f"Unknown model_type {cfg.model_type!r}. Choose 'cyclegan' or 'not'.")


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    dataset = UnpairedImageDataset(
        cfg.data.root_a,
        cfg.data.root_b,
        cfg.data.image_size,
        extra_roots_a=list(cfg.data.extra_roots_a),
        extra_roots_b=list(cfg.data.extra_roots_b),
        align_a=cfg.data.align_a,
        anime_offset_x=cfg.data.anime_offset_x,
        anime_offset_y=cfg.data.anime_offset_y,
    )

    train_dl = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = _make_model(cfg)

    if cfg.model_type == "not":
        logger = MLFlowLogger(
            experiment_name="kaoanime-not",
            tracking_uri=cfg.not_.mlflow_tracking_uri,
        )
        # Lightning increments global_step once per optimizer.step() call in manual
        # optimization mode, not once per training_step. With t_iters+1 optimizer
        # steps per batch, multiply so not_.max_steps means "batch iterations".
        _lightning_max_steps = cfg.not_.max_steps * (cfg.not_.t_iters + 1)
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_steps=_lightning_max_steps,
            max_epochs=-1,
            precision=cfg.not_.precision,
            log_every_n_steps=cfg.not_.log_every_n_steps,
            logger=logger,
            callbacks=[ModelCheckpoint(filename="step{step:06d}", save_last=True, save_top_k=0)],
        )
    else:
        logger = MLFlowLogger(
            experiment_name="kaoanime",
            tracking_uri=cfg.train.mlflow_tracking_uri,
        )
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_epochs=cfg.train.max_epochs,
            precision=cfg.train.precision,
            log_every_n_steps=cfg.train.log_every_n_steps,
            logger=logger,
            callbacks=[ModelCheckpoint(filename="epoch{epoch:03d}", save_last=True, save_top_k=0)],
        )

    trainer.fit(model, train_dl)

    ckpt_path = trainer.checkpoint_callback.last_model_path
    if ckpt_path:
        logger.experiment.log_artifact(logger.run_id, ckpt_path)


if __name__ == "__main__":
    main()
