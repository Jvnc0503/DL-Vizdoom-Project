from __future__ import annotations

import importlib
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class FrozenVisionFrontend(nn.Module):
    def __init__(
        self,
        mobilenet_model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        yolo_model_name: str = "yolo26s.pt",
        yolo_imgsz: int = 320,
        yolo_conf: float = 0.25,
        yolo_max_det: int = 10,
        use_yolo: bool = True,
    ) -> None:
        super().__init__()
        try:
            timm = importlib.import_module("timm")
        except Exception as exc:
            raise ImportError("timm no está instalado. Instala con `pip install timm`.") from exc

        self.mobilenet_model_name = str(mobilenet_model_name)
        raw_yolo_model_name = str(yolo_model_name)
        if os.path.isabs(raw_yolo_model_name):
            resolved_yolo_model_name = raw_yolo_model_name
        else:
            resolved_yolo_model_name = os.path.join(os.path.dirname(__file__), raw_yolo_model_name)
        self.yolo_model_name = resolved_yolo_model_name
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_conf = float(yolo_conf)
        self.yolo_max_det = int(yolo_max_det)
        self.use_yolo = bool(use_yolo)

        self.mobilenet = timm.create_model(
            self.mobilenet_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.mobilenet.eval()
        for p in self.mobilenet.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
            mobilenet_dim = int(self.mobilenet(dummy).shape[-1])

        self._yolo_model: Optional[Any] = None
        self.yolo_feature_dim = 8 if self.use_yolo else 0

        if self.use_yolo:
            try:
                ultralytics = importlib.import_module("ultralytics")
            except Exception as exc:
                raise ImportError("ultralytics no está instalado. Instala con `pip install ultralytics`.") from exc
            YOLO = ultralytics.YOLO
            object.__setattr__(self, "_yolo_model", YOLO(self.yolo_model_name))

        self.output_dim = mobilenet_dim + self.yolo_feature_dim

    def _torch_device_for_yolo(self, device: torch.device) -> str:
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            return str(idx)
        return "cpu"

    def _extract_yolo_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_yolo or self._yolo_model is None:
            return torch.zeros((x.shape[0], 0), dtype=x.dtype, device=x.device)

        b = x.shape[0]
        x_uint8 = (
            x.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        images = [np.asarray(img, dtype=np.uint8) for img in x_uint8]

        results = self._yolo_model.predict(
            source=images,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            max_det=self.yolo_max_det,
            device=self._torch_device_for_yolo(x.device),
            verbose=False,
        )

        feat = torch.zeros((b, self.yolo_feature_dim), dtype=torch.float32, device=x.device)
        for i, res in enumerate(results):
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.xywhn is None or boxes.xywhn.numel() == 0:
                continue

            xywhn = boxes.xywhn.detach().to(x.device, dtype=torch.float32)
            conf = boxes.conf.detach().to(x.device, dtype=torch.float32)
            n = min(int(xywhn.shape[0]), max(1, self.yolo_max_det))
            xywhn_n = xywhn[:n]
            conf_n = conf[:n]

            cx = xywhn_n[:, 0]
            cy = xywhn_n[:, 1]
            w = xywhn_n[:, 2]
            h = xywhn_n[:, 3]
            area = w * h

            feat[i, 0] = float(n) / float(max(1, self.yolo_max_det))
            feat[i, 1] = conf_n.mean()
            feat[i, 2] = conf_n.max()
            feat[i, 3] = cx.mean()
            feat[i, 4] = cy.mean()
            feat[i, 5] = w.mean()
            feat[i, 6] = h.mean()
            feat[i, 7] = area.mean()

        return feat.to(dtype=x.dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)

        self.mobilenet.eval()
        z_global = self.mobilenet(x)

        z_obj = self._extract_yolo_features(x)
        if z_obj.numel() == 0:
            return z_global

        return torch.cat([z_global, z_obj], dim=1)
