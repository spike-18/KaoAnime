# train.py
import hydra
import lightning as pl

from kaoanime.config import Config, register_configs
from kaoanime.model import KaoAnimeModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader, get_transforms

register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    transform = get_transforms("train", image_size=cfg.data.image_size)
    dataset = UnpairedImageDataset(cfg.data.root_a, cfg.data.root_b, transform)
    train_dl = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    model = KaoAnimeModel(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )
    trainer.fit(model, train_dl)


if __name__ == "__main__":
    main()
