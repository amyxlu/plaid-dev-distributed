import torch
from pathlib import Path
from .config import ModelConfig, dataclass_from_dict, make_denoiser_wrapper, make_model
import json


def infer_ckpt_path(model_dir, model_id, model_step=-1):
    model_dir = Path(model_dir)
    if (model_step == -1) or (model_step is None):
        state_path = model_dir / "checkpoints" / model_id / "state.json"
        state = json.load(open(state_path))
        return state["latest_checkpoint"]
    else:
        return str(
            model_dir
            / "checkpoints"
            / model_id
            / f"{model_step:08}.pth"
        )


def load_model(filename, use_ema=True):
    filename = Path(filename)
    print("Loading checkpoint from", filename)
    
    ckpt = torch.load(filename, map_location="cpu")
    max_seq_len = ckpt["config"]["max_seq_len"]
    model_config = dataclass_from_dict(ModelConfig, ckpt['config']['model_config'])
    inner_model = (
        make_model(model_config, max_seq_len=int(max_seq_len)).eval().requires_grad_(False)
    )
    if use_ema:
        inner_model.load_state_dict(ckpt["model_ema"])
    else:
        inner_model.load_state_dict(ckpt["model"])
    model = make_denoiser_wrapper(model_config)(inner_model)
    return model, inner_model, model_config

