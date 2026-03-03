from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from bc.model import BCPolicyNet
from doom_controller import DoomController


def _preprocess_screen(screen_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(screen_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def _resolve_conflicts(action: np.ndarray, probs: np.ndarray, button_names: List[str]) -> np.ndarray:
    out = action.copy()

    def suppress_pair(a: str, b: str) -> None:
        if a in button_names and b in button_names:
            ia = button_names.index(a)
            ib = button_names.index(b)
            if out[ia] == 1 and out[ib] == 1:
                if probs[ia] >= probs[ib]:
                    out[ib] = 0
                else:
                    out[ia] = 0

    suppress_pair("MOVE_FORWARD", "MOVE_BACKWARD")
    suppress_pair("MOVE_LEFT", "MOVE_RIGHT")
    suppress_pair("TURN_LEFT", "TURN_RIGHT")

    weapon_buttons = [b for b in button_names if b.startswith("SELECT_WEAPON")]
    active_weapon_idx = [button_names.index(b) for b in weapon_buttons if out[button_names.index(b)] == 1]
    if len(active_weapon_idx) > 1:
        best_i = max(active_weapon_idx, key=lambda idx: probs[idx])
        for idx in active_weapon_idx:
            out[idx] = 1 if idx == best_i else 0

    return out


def _infer_action(
    model: BCPolicyNet,
    obs_screen: np.ndarray,
    image_size: int,
    button_names: List[str],
    threshold: float,
    action_thresholds: Optional[Sequence[float]],
    device: torch.device,
    stochastic: bool,
) -> np.ndarray:
    x = _preprocess_screen(obs_screen, image_size=image_size).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    if stochastic:
        action = (np.random.rand(probs.shape[0]) < probs).astype(np.int32)
    else:
        if action_thresholds is not None and len(action_thresholds) == probs.shape[0]:
            thr = np.asarray(action_thresholds, dtype=np.float32)
            action = (probs >= thr).astype(np.int32)
        else:
            action = (probs >= threshold).astype(np.int32)

    action = _resolve_conflicts(action, probs, button_names)
    return action


def _terminal_reason(ctrl: DoomController, truncated: bool, obs: dict) -> str:
    if truncated:
        return "timeout"
    try:
        is_dead = bool(ctrl.game.is_player_dead())
    except Exception:
        is_dead = False
    if is_dead:
        return "death"
    gv = obs.get("gamevariables")
    if isinstance(gv, np.ndarray) and "HEALTH" in ctrl.game_variable_names:
        h_idx = ctrl.game_variable_names.index("HEALTH")
        if float(gv[h_idx]) <= 0:
            return "death"
    return "success"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecutar agente BC entrenado en VizDoom")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta a best.pt o last.pt")
    parser.add_argument("--config", type=str, default="game_config.yaml")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral de acción (default: el del checkpoint)")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = sin límite")
    parser.add_argument("--repeat", type=int, default=1, help="Tics por acción")
    parser.add_argument("--target-hz", type=float, default=35.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stochastic", action="store_true", help="Si se activa, samplea Bernoulli en vez de threshold")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu")
    button_names: List[str] = list(payload["button_names"])
    image_size: int = int(payload.get("image_size", 128))
    threshold: float = float(payload.get("threshold", 0.5) if args.threshold is None else args.threshold)
    action_thresholds: Optional[List[float]] = None
    if args.threshold is None:
        loaded = payload.get("action_thresholds", None)
        if isinstance(loaded, list) and len(loaded) == len(button_names):
            action_thresholds = [float(t) for t in loaded]

    model = BCPolicyNet(num_actions=len(button_names))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    ctrl = DoomController(config_path=args.config)
    obs = ctrl.reset()

    runtime_button_names = ctrl.button_names
    if runtime_button_names != button_names:
        ctrl.close()
        raise RuntimeError(
            "button_names del checkpoint no coinciden con DoomController config. "
            "Entrena y ejecuta con el mismo orden de controles."
        )

    period = 1.0 / max(1e-6, float(args.target_hz))
    next_t = time.perf_counter()
    total_reward = 0.0
    steps = 0
    term_reason: Optional[str] = None

    try:
        while True:
            if args.max_steps > 0 and steps >= args.max_steps:
                term_reason = "max_steps"
                break

            action = _infer_action(
                model=model,
                obs_screen=obs["screen"],
                image_size=image_size,
                button_names=button_names,
                threshold=threshold,
                action_thresholds=action_thresholds,
                device=device,
                stochastic=bool(args.stochastic),
            )

            obs, reward, terminated, truncated, _ = ctrl.step(action, repeat=max(1, int(args.repeat)))
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                term_reason = _terminal_reason(ctrl, truncated, obs)
                break

            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    finally:
        ctrl.close()

    print("\n" + "=" * 60)
    print(f"Agente BC finalizado | reason={term_reason} | steps={steps} | reward={total_reward:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
