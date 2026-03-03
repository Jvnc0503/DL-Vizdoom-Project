from __future__ import annotations

import torch
import torch.nn as nn

from common.vision_frontend import FrozenVisionFrontend


class BCPolicyNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        mobilenet_model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        yolo_model_name: str = "yolo26s.pt",
        yolo_imgsz: int = 320,
        yolo_conf: float = 0.25,
        yolo_max_det: int = 10,
        use_yolo: bool = True,
    ) -> None:
        super().__init__()
        self.vision_frontend = FrozenVisionFrontend(
            mobilenet_model_name=mobilenet_model_name,
            yolo_model_name=yolo_model_name,
            yolo_imgsz=yolo_imgsz,
            yolo_conf=yolo_conf,
            yolo_max_det=yolo_max_det,
            use_yolo=use_yolo,
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.vision_frontend.output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_frontend = self.vision_frontend(x)
        z = self.encoder(z_frontend)
        logits = self.actor_head(z)
        return logits
