# infer.py
from pathlib import Path

import torch
import hydra

from kaoanime.config import Config, register_configs
from kaoanime.data import ensure_data
from kaoanime.inference import run_inference
from kaoanime.model_cyclegan import KaoAnimeModel
from kaoanime.model_not import NOTModel
from kaoanime.model_store import CKPT_NAME, ensure_model

register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    if not cfg.eval.input:
        raise ValueError(
            "eval.input must be specified, e.g.: eval.input=data/selfie2anime/testA"
        )

    ensure_data(cfg)
    ensure_model(cfg)
    checkpoint = cfg.eval.checkpoint or str(Path(cfg.eval.model_dir) / CKPT_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = NOTModel if cfg.model_type == "not" else KaoAnimeModel
    model = cls.load_from_checkpoint(
        checkpoint, cfg=cfg, map_location=device, strict=True
    )
    model.to(device).float()
    model.eval()

    written = run_inference(
        model,
        input_path=cfg.eval.input,
        output_dir=cfg.eval.output_dir,
        image_size=cfg.data.image_size,
        direction=cfg.eval.direction,
        device=device,
        align=cfg.eval.align,
    )
    print(f"Saved {len(written)} image(s) to {cfg.eval.output_dir}")


if __name__ == "__main__":
    main()
