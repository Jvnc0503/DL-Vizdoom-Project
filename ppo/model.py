from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from common.vision_frontend import FrozenVisionFrontend


class PPOActorCritic(nn.Module):
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
        self.num_actions = int(num_actions)

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
            nn.Linear(256, self.num_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_frontend = self.vision_frontend(x)
        return self.encoder(z_frontend)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_dist(self, logits: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=logits)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logits, value = self.forward(x)
        dist = self.get_dist(logits)

        if action is None:
            if deterministic:
                action = (torch.sigmoid(logits) >= 0.5).float()
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)

        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
        }


def load_bc_weights_into_ppo(ppo_model: PPOActorCritic, bc_state_dict: Dict[str, torch.Tensor]) -> None:
    mapped: Dict[str, torch.Tensor] = {}

    for key, tensor in bc_state_dict.items():
        if key.startswith("vision_frontend."):
            mapped[key] = tensor
            continue

        if key.startswith("encoder."):
            mapped[key] = tensor
            continue

        if key.startswith("actor_head."):
            suffix = key[len("actor_head.") :]
            mapped[f"actor_head.{suffix}"] = tensor

    ppo_model.load_state_dict(mapped, strict=False)
