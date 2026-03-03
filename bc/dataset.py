from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SessionInfo:
    session_dir: str
    meta_parquet: str
    video_path: str
    button_names: List[str]
    num_steps: int


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON inválido (se esperaba dict): {path}")
    return payload


def _parse_action_bin(value: Any, expected_len: int) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        arr = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value).reshape(-1)
    elif isinstance(value, str):
        value = value.strip()
        parsed: Any
        try:
            parsed = json.loads(value)
        except Exception:
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return None
        if not isinstance(parsed, (list, tuple)):
            return None
        arr = np.asarray(parsed).reshape(-1)
    else:
        return None

    if arr.size != expected_len:
        return None
    arr = (arr.astype(np.int32) != 0).astype(np.float32)
    return arr


def _safe_read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def discover_sessions(recordings_dir: str, min_steps: int = 64) -> List[SessionInfo]:
    sessions: List[SessionInfo] = []
    if not os.path.isdir(recordings_dir):
        raise FileNotFoundError(f"No existe recordings_dir: {recordings_dir}")

    for name in sorted(os.listdir(recordings_dir)):
        session_dir = os.path.join(recordings_dir, name)
        if not os.path.isdir(session_dir):
            continue

        meta_json_path = os.path.join(session_dir, "session_meta.json")
        meta_parquet = os.path.join(session_dir, "meta.parquet")
        if not (os.path.isfile(meta_json_path) and os.path.isfile(meta_parquet)):
            continue

        try:
            meta_json = _load_json(meta_json_path)
            button_names = list(meta_json.get("button_names") or meta_json.get("buttons") or [])
            video_rel = str(meta_json.get("video_path", "screen.mkv"))
            video_path = os.path.join(session_dir, video_rel)
            if not button_names:
                continue
            if not os.path.isfile(video_path):
                fallback = os.path.join(session_dir, "screen.mkv")
                if os.path.isfile(fallback):
                    video_path = fallback
                else:
                    continue

            num_steps = int(meta_json.get("num_steps", 0))
            if num_steps < min_steps:
                continue

            sessions.append(
                SessionInfo(
                    session_dir=session_dir,
                    meta_parquet=meta_parquet,
                    video_path=video_path,
                    button_names=button_names,
                    num_steps=num_steps,
                )
            )
        except Exception:
            continue

    if not sessions:
        raise RuntimeError(
            "No se encontraron sesiones válidas. Asegúrate de tener grabaciones en recordings/session_*/"
        )
    return sessions


def split_sessions(sessions: Sequence[SessionInfo], val_ratio: float, seed: int) -> Tuple[List[SessionInfo], List[SessionInfo]]:
    if not sessions:
        raise ValueError("No hay sesiones para dividir.")

    sess = list(sessions)
    rng = random.Random(seed)
    rng.shuffle(sess)

    n_total = len(sess)
    n_val = int(round(n_total * float(val_ratio)))
    n_val = max(1, min(n_total - 1, n_val)) if n_total > 1 else 0
    val_sessions = sess[:n_val]
    train_sessions = sess[n_val:] if n_val > 0 else sess
    return train_sessions, val_sessions


class BCDataset(Dataset):
    def __init__(self, sessions: Sequence[SessionInfo], image_size: int = 128) -> None:
        if not sessions:
            raise ValueError("BCDataset requiere al menos una sesión.")

        self.sessions: List[SessionInfo] = list(sessions)
        self.image_size = int(image_size)
        self.button_names = list(self.sessions[0].button_names)
        self.num_actions = len(self.button_names)

        for s in self.sessions[1:]:
            if s.button_names != self.button_names:
                raise ValueError("Todas las sesiones deben usar el mismo orden de button_names.")

        self.samples: List[Tuple[int, int, np.ndarray]] = []
        for session_index, session in enumerate(self.sessions):
            df = _safe_read_parquet(session.meta_parquet)
            if "action_bin" not in df.columns:
                continue

            action_list = df["action_bin"].tolist()
            cap = cv2.VideoCapture(session.video_path)
            if not cap.isOpened():
                cap.release()
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            max_rows = min(len(action_list), max(0, frame_count))
            for frame_idx in range(max_rows):
                action = _parse_action_bin(action_list[frame_idx], self.num_actions)
                if action is None:
                    continue
                self.samples.append((session_index, frame_idx, action))

        if not self.samples:
            raise RuntimeError("No se pudieron construir muestras (video/action_bin).")

        self._caps: Dict[int, cv2.VideoCapture] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cap(self, session_index: int) -> cv2.VideoCapture:
        cap = self._caps.get(session_index)
        if cap is None or (not cap.isOpened()):
            video_path = self.sessions[session_index].video_path
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir video: {video_path}")
            self._caps[session_index] = cap
        return cap

    def _read_frame_rgb(self, session_index: int, frame_idx: int) -> np.ndarray:
        cap = self._get_cap(session_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"No se pudo leer frame {frame_idx} de sesión {self.sessions[session_index].session_dir}"
            )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_index, frame_idx, action = self.samples[idx]
        frame_rgb = self._read_frame_rgb(session_index, frame_idx)
        frame_resized = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        x = frame_resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        y = action.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def close(self) -> None:
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps = {}
