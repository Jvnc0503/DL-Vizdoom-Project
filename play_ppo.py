from __future__ import annotations

import argparse
import time
from typing import List, Optional

import cv2
import numpy as np
import torch

from doom_controller import DoomController
from ppo.model import PPOActorCritic


def preprocess_screen(screen_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(screen_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def terminal_reason(ctrl: DoomController, truncated: bool, obs: dict) -> str:
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
    parser = argparse.ArgumentParser(description="Ejecutar agente PPO entrenado en VizDoom")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta a checkpoint PPO (.pt)")
    parser.add_argument("--config", type=str, default="game_config.yaml")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = sin límite")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--target-hz", type=float, default=35.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--deterministic", action="store_true", help="Usar política determinística (threshold 0.5)")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu")
    button_names: List[str] = list(payload["button_names"])
    image_size = int(payload.get("image_size", 128))

    model = PPOActorCritic(num_actions=len(button_names))
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
    reason: Optional[str] = None

    try:
        while True:
            if args.max_steps > 0 and steps >= args.max_steps:
                reason = "max_steps"
                break

            x = preprocess_screen(obs["screen"], image_size=image_size).to(device)
            with torch.no_grad():
                out = model.get_action_and_value(x, deterministic=bool(args.deterministic))

            action = out["action"].squeeze(0).cpu().numpy().astype(np.int32)
            obs, reward, terminated, truncated, _ = ctrl.step(action, repeat=max(1, int(args.repeat)))

            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                reason = terminal_reason(ctrl, truncated, obs)
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
    print(f"Agente PPO finalizado | reason={reason} | steps={steps} | reward={total_reward:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
