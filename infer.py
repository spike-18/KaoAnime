# infer.py
import hydra
import torch

from kaoanime.config import Config, register_configs
from kaoanime.data import ensure_data
from kaoanime.inference import run_inference
from kaoanime.model_cyclegan import KaoAnimeModel
from kaoanime.model_not import NOTModel

register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    if not cfg.eval.input:
        raise ValueError(
            "eval.input must be specified, e.g.: eval.input=data/demo/testA"
        )
    if not cfg.eval.checkpoint:
        raise ValueError(
            "eval.checkpoint must point to a model checkpoint. Download the "
            "published weights first (uv run python scripts/download_model.py), "
            "then e.g.: eval.checkpoint=models/export/NOT.ckpt"
        )

    ensure_data(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = NOTModel if cfg.model_type == "not" else KaoAnimeModel
    model = cls.load_from_checkpoint(
        cfg.eval.checkpoint, cfg=cfg, map_location=device, strict=True
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
