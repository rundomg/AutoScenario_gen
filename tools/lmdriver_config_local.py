"""Local LMDrive checkpoint configuration for the converted scenario."""

import os


class GlobalConfig:
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40
    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40
    max_throttle = 0.75
    brake_speed = 0.1
    brake_ratio = 1.1
    clip_delta = 0.35

    llm_model = os.environ.get(
        "LMDRIVE_LLM_MODEL", "/home/zx/code/LMDrive/data/llava-v1.5-7b"
    )
    preception_model = "memfuser_baseline_e1d3_return_feature"
    preception_model_ckpt = os.environ.get(
        "LMDRIVE_PERCEPTION_CKPT",
        "/home/zx/code/LMDrive/vision-encoder-r50.pth.tar",
    )
    lmdrive_ckpt = os.environ.get(
        "LMDRIVE_CKPT", "/home/zx/code/LMDrive/llava-v1.5-checkpoint.pth"
    )
    agent_use_notice = True
    sample_rate = 2

    def __init__(self, **kwargs):
        missing = [
            path
            for path in (self.llm_model, self.preception_model_ckpt, self.lmdrive_ckpt)
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError("LMDrive model path(s) not found: {}".format(", ".join(missing)))
        for key, value in kwargs.items():
            setattr(self, key, value)
