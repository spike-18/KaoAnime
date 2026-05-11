# infer.py
import torch
import hydra

from kaoanime.config import Config, register_configs
from kaoanime.inference import run_inference
from kaoanime.model import KaoAnimeModel

register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    if not cfg.eval.checkpoint:
        raise ValueError("eval.checkpoint must be specified, e.g.: eval.checkpoint=outputs/.../last.ckpt")
    if not cfg.eval.input:
        raise ValueError("eval.input must be specified, e.g.: eval.input=data/selfie2anime/testA")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KaoAnimeModel.load_from_checkpoint(cfg.eval.checkpoint, cfg=cfg, map_location=device)
    model.to(device).float()
    model.eval()

    written = run_inference(
        model,
        input_path=cfg.eval.input,
        output_dir=cfg.eval.output_dir,
        image_size=cfg.data.image_size,
        direction=cfg.eval.direction,
        device=device,
    )
    print(f"Saved {len(written)} image(s) to {cfg.eval.output_dir}")


if __name__ == "__main__":
    main()
