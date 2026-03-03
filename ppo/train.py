from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from doom_controller import DoomController
from ppo.model import PPOActorCritic, load_bc_weights_into_ppo


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_screen(screen_rgb: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(screen_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    return x_t


def gamevariables_to_dict(obs: Dict[str, Any], gv_names: list[str]) -> Dict[str, float]:
    gv = obs.get("gamevariables", None)
    if not isinstance(gv, np.ndarray):
        return {}
    vals = gv.reshape(-1).tolist()
    return {name: float(val) for name, val in zip(gv_names, vals)}


def build_run_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"ppo_run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _build_scenario_config_list(primary_config: str, scenario_configs_dir: str) -> list[str]:
    if not str(scenario_configs_dir).strip():
        return [os.path.abspath(str(primary_config))]

    abs_dir = os.path.abspath(str(scenario_configs_dir))
    if not os.path.isdir(abs_dir):
        raise NotADirectoryError(f"--scenario-configs debe apuntar a una carpeta válida: {abs_dir}")

    yaml_paths = [
        os.path.join(abs_dir, name)
        for name in sorted(os.listdir(abs_dir))
        if os.path.isfile(os.path.join(abs_dir, name)) and name.lower().endswith((".yaml", ".yml"))
    ]
    if not yaml_paths:
        raise FileNotFoundError(f"No se encontraron archivos .yaml/.yml en: {abs_dir}")

    return [os.path.abspath(path) for path in yaml_paths]


def _scenario_index_from_step(global_step: int, switch_every_timesteps: int, num_scenarios: int) -> int:
    if num_scenarios <= 1 or int(switch_every_timesteps) <= 0:
        return 0
    block = max(0, int(global_step)) // max(1, int(switch_every_timesteps))
    return int(block % num_scenarios)


def _extract_reward_shaping_params(ctrl: DoomController) -> Dict[str, float]:
    rew_cfg = ctrl.cfg.get("reward", {}) if isinstance(ctrl.cfg, dict) else {}
    return {
        "kill_reward": float(rew_cfg.get("kill_reward", 0.0)),
        "pickup_health": float(rew_cfg.get("pickup_health", 0.0)),
        "pickup_armor": float(rew_cfg.get("pickup_armor", 0.0)),
        "level_clear_bonus": float(rew_cfg.get("level_clear_bonus", 0.0)),
        "ammo_spend_penalty": float(rew_cfg.get("ammo_spend_penalty", 0.0)),
        "damage_taken_penalty": float(rew_cfg.get("damage_taken_penalty", 0.0)),
        "armor_damage_penalty": float(rew_cfg.get("armor_damage_penalty", 0.0)),
    }


def _find_latest_last_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None

    run_dirs = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("ppo_run_") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not run_dirs:
        return None

    run_dirs.sort(reverse=True)
    for run_dir in run_dirs:
        candidate = os.path.join(run_dir, "last.pt")
        if os.path.isfile(candidate):
            return candidate
    return None


def _run_dir_from_checkpoint_path(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    folder = os.path.dirname(checkpoint_path)
    if os.path.basename(folder) == "checkpoints":
        return os.path.dirname(folder)
    return folder


def set_encoder_trainable(model: PPOActorCritic, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)


def terminal_reason_from_step(
    ctrl: DoomController,
    truncated: bool,
    current_gv: Dict[str, float],
) -> str:
    if truncated:
        return "timeout"

    try:
        if bool(ctrl.game.is_player_dead()):
            return "death"
    except Exception:
        pass

    health_val = current_gv.get("HEALTH", None)
    if health_val is not None and health_val <= 0:
        return "death"
    return "success"


def compute_transition_shaping(
    prev_gv: Dict[str, float],
    current_gv: Dict[str, float],
    kill_reward: float,
    pickup_health_reward: float,
    pickup_armor_reward: float,
    ammo_spend_penalty: float,
    damage_taken_penalty: float,
    armor_damage_penalty: float,
) -> float:
    shaping_extra = 0.0
    if not prev_gv or not current_gv:
        return shaping_extra

    if "KILLCOUNT" in current_gv and "KILLCOUNT" in prev_gv:
        delta_k = int(current_gv.get("KILLCOUNT", 0) - prev_gv.get("KILLCOUNT", 0))
        if delta_k > 0:
            shaping_extra += float(delta_k) * kill_reward

    if "HEALTH" in current_gv and "HEALTH" in prev_gv:
        delta_h = float(current_gv.get("HEALTH", 0.0) - prev_gv.get("HEALTH", 0.0))
        if delta_h > 0:
            shaping_extra += (delta_h / 25.0) * pickup_health_reward

    if "ARMOR" in current_gv and "ARMOR" in prev_gv:
        delta_a = float(current_gv.get("ARMOR", 0.0) - prev_gv.get("ARMOR", 0.0))
        if delta_a > 0:
            shaping_extra += (delta_a / 25.0) * pickup_armor_reward

    ammo_used = 0.0
    for ammo_key in ("AMMO1", "AMMO2", "AMMO3", "AMMO4"):
        if ammo_key in current_gv and ammo_key in prev_gv:
            ammo_used += max(0.0, float(prev_gv.get(ammo_key, 0.0) - current_gv.get(ammo_key, 0.0)))
    if ammo_used > 0.0:
        shaping_extra += ammo_used * ammo_spend_penalty

    if "HEALTH" in current_gv and "HEALTH" in prev_gv:
        health_lost = max(0.0, float(prev_gv.get("HEALTH", 0.0) - current_gv.get("HEALTH", 0.0)))
        if health_lost > 0.0:
            shaping_extra += health_lost * damage_taken_penalty

    if "ARMOR" in current_gv and "ARMOR" in prev_gv:
        armor_lost = max(0.0, float(prev_gv.get("ARMOR", 0.0) - current_gv.get("ARMOR", 0.0)))
        if armor_lost > 0.0:
            shaping_extra += armor_lost * armor_damage_penalty

    return shaping_extra


@torch.no_grad()
def evaluate_policy(
    model: PPOActorCritic,
    ctrl: DoomController,
    device: torch.device,
    image_size: int,
    repeat: int,
    num_episodes: int,
    max_steps_per_episode: int,
    gv_names: list[str],
    use_reward_shaping: bool,
    kill_reward: float,
    pickup_health_reward: float,
    pickup_armor_reward: float,
    level_clear_bonus: float,
    ammo_spend_penalty: float,
    damage_taken_penalty: float,
    armor_damage_penalty: float,
) -> Dict[str, float]:
    model.eval()

    ep_rewards: list[float] = []
    ep_lens: list[int] = []

    for _ in range(max(1, int(num_episodes))):
        obs_np = ctrl.reset()
        prev_gv = gamevariables_to_dict(obs_np, gv_names)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done and ep_len < max(1, int(max_steps_per_episode)):
            obs_t = preprocess_screen(obs_np["screen"], image_size, device)
            out = model.get_action_and_value(obs_t, deterministic=True)
            action_np = out["action"].squeeze(0).detach().cpu().numpy().astype(np.int32)

            obs_next, reward, terminated, truncated, _ = ctrl.step(action_np, repeat=max(1, int(repeat)))
            current_gv = gamevariables_to_dict(obs_next, gv_names)

            if use_reward_shaping:
                reward = float(reward) + compute_transition_shaping(
                    prev_gv=prev_gv,
                    current_gv=current_gv,
                    kill_reward=kill_reward,
                    pickup_health_reward=pickup_health_reward,
                    pickup_armor_reward=pickup_armor_reward,
                    ammo_spend_penalty=ammo_spend_penalty,
                    damage_taken_penalty=damage_taken_penalty,
                    armor_damage_penalty=armor_damage_penalty,
                )

            done = bool(terminated or truncated)
            if done and use_reward_shaping:
                reason = terminal_reason_from_step(ctrl, bool(truncated), current_gv)
                if reason == "success" and level_clear_bonus != 0.0:
                    reward = float(reward) + float(level_clear_bonus)

            ep_reward += float(reward)
            ep_len += 1

            obs_np = obs_next
            if current_gv:
                prev_gv = current_gv

        ep_rewards.append(float(ep_reward))
        ep_lens.append(int(ep_len))

    return {
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "mean_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO fine-tuning para VizDoom (inicializable desde BC)")
    parser.add_argument("--config", type=str, default="game_config.yaml")
    parser.add_argument(
        "--scenario-configs",
        type=str,
        default="",
        help="Carpeta con configs .yaml/.yml para rotar escenarios durante entrenamiento (round-robin).",
    )
    parser.add_argument(
        "--switch-every-timesteps",
        type=int,
        default=0,
        help="Si >0, rota al siguiente scenario config cada N timesteps globales.",
    )
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--bc-checkpoint", type=str, default="", help="Checkpoint BC para inicializar actor/encoder.")
    parser.add_argument("--resume-from", type=str, default="", help="Checkpoint PPO para continuar entrenamiento.")
    parser.add_argument("--resume-latest", action="store_true", help="Reanudar desde el último ppo_run_*/last.pt")

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=1024, help="Longitud de rollout por update")
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1, help="Tics por acción")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--mobilenet-model-name", type=str, default="mobilenetv4_conv_small.e2400_r224_in1k")
    parser.add_argument("--yolo-model-name", type=str, default="yolo26s.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=320)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-max-det", type=int, default=10)
    parser.add_argument("--disable-yolo", action="store_true")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--anneal-lr", action="store_true", help="Decaer LR linealmente con el progreso.")

    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=2000)
    parser.add_argument("--freeze-encoder-updates", type=int, default=0)
    parser.add_argument(
        "--disable-reward-shaping",
        action="store_true",
        help="Si se activa, usa solo reward del entorno (sin shaping adicional en Python).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenario_configs = _build_scenario_config_list(args.config, args.scenario_configs)

    ctrl = DoomController(config_path=scenario_configs[0])
    button_names = ctrl.button_names
    num_actions = len(button_names)
    gv_names = ctrl.game_variable_names

    for cfg_path in scenario_configs[1:]:
        probe_ctrl = DoomController(config_path=cfg_path)
        try:
            if probe_ctrl.button_names != button_names:
                ctrl.close()
                raise RuntimeError(
                    f"button_names incompatibles entre escenarios: {scenario_configs[0]} vs {cfg_path}"
                )
            if probe_ctrl.game_variable_names != gv_names:
                ctrl.close()
                raise RuntimeError(
                    f"game_variable_names incompatibles entre escenarios: {scenario_configs[0]} vs {cfg_path}"
                )
        finally:
            probe_ctrl.close()

    reward_params = _extract_reward_shaping_params(ctrl)
    kill_reward = float(reward_params["kill_reward"])
    pickup_health_reward = float(reward_params["pickup_health"])
    pickup_armor_reward = float(reward_params["pickup_armor"])
    level_clear_bonus = float(reward_params["level_clear_bonus"])
    ammo_spend_penalty = float(reward_params["ammo_spend_penalty"])
    damage_taken_penalty = float(reward_params["damage_taken_penalty"])
    armor_damage_penalty = float(reward_params["armor_damage_penalty"])
    use_reward_shaping = not bool(args.disable_reward_shaping)

    model = PPOActorCritic(
        num_actions=num_actions,
        mobilenet_model_name=args.mobilenet_model_name,
        yolo_model_name=args.yolo_model_name,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        yolo_max_det=args.yolo_max_det,
        use_yolo=not bool(args.disable_yolo),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    resume_path = ""
    if bool(args.resume_latest):
        latest = _find_latest_last_checkpoint(args.output_dir)
        if latest is None:
            ctrl.close()
            raise FileNotFoundError(f"No se encontró checkpoint PPO en: {args.output_dir}")
        resume_path = latest
    elif args.resume_from:
        resume_path = os.path.abspath(args.resume_from)

    if resume_path:
        if not os.path.isfile(resume_path):
            ctrl.close()
            raise FileNotFoundError(f"Checkpoint PPO no encontrado: {resume_path}")
        run_dir = _run_dir_from_checkpoint_path(resume_path)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = build_run_dir(args.output_dir)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    config_path = os.path.join(run_dir, "training_config.json")

    global_step = 0
    start_update = 1
    best_mean_ep_reward = -float("inf")
    best_eval_reward = -float("inf")

    if resume_path:
        payload: Dict[str, Any] = torch.load(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        ckpt_buttons = list(payload.get("button_names", []))
        if ckpt_buttons and ckpt_buttons != button_names:
            ctrl.close()
            raise RuntimeError("button_names del checkpoint PPO no coinciden con config actual.")

        global_step = int(payload.get("global_step", 0))
        start_update = int(payload.get("update", 0)) + 1
        best_mean_ep_reward = float(payload.get("best_mean_ep_reward", -float("inf")))
        best_eval_reward = float(payload.get("best_eval_reward", best_mean_ep_reward))
        print(f"[INFO] Reanudando PPO desde: {resume_path}")

    elif args.bc_checkpoint:
        bc_path = os.path.abspath(args.bc_checkpoint)
        if not os.path.isfile(bc_path):
            ctrl.close()
            raise FileNotFoundError(f"Checkpoint BC no encontrado: {bc_path}")
        bc_payload: Dict[str, Any] = torch.load(bc_path, map_location="cpu")
        bc_buttons = list(bc_payload.get("button_names", []))
        if bc_buttons and bc_buttons != button_names:
            ctrl.close()
            raise RuntimeError("button_names del checkpoint BC no coinciden con config actual.")

        load_bc_weights_into_ppo(model, bc_payload["model_state_dict"])
        print(f"[INFO] Inicialización PPO desde BC: {bc_path}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "scenario_configs_dir": os.path.abspath(args.scenario_configs) if args.scenario_configs else None,
                "scenario_configs": scenario_configs,
                "switch_every_timesteps": int(args.switch_every_timesteps),
                "output_dir": os.path.abspath(args.output_dir),
                "run_dir": os.path.abspath(run_dir),
                "button_names": button_names,
                "num_actions": num_actions,
                "seed": int(args.seed),
                "device": device.type,
                "total_timesteps": int(args.total_timesteps),
                "num_steps": int(args.num_steps),
                "update_epochs": int(args.update_epochs),
                "minibatch_size": int(args.minibatch_size),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "clip_coef": float(args.clip_coef),
                "ent_coef": float(args.ent_coef),
                "vf_coef": float(args.vf_coef),
                "max_grad_norm": float(args.max_grad_norm),
                "target_kl": float(args.target_kl),
                "learning_rate": float(args.learning_rate),
                "anneal_lr": bool(args.anneal_lr),
                "repeat": int(args.repeat),
                "image_size": int(args.image_size),
                "mobilenet_model_name": str(args.mobilenet_model_name),
                "yolo_model_name": str(args.yolo_model_name),
                "yolo_imgsz": int(args.yolo_imgsz),
                "yolo_conf": float(args.yolo_conf),
                "yolo_max_det": int(args.yolo_max_det),
                "use_yolo": bool(not args.disable_yolo),
                "eval_every_updates": int(args.eval_every_updates),
                "eval_episodes": int(args.eval_episodes),
                "eval_max_steps": int(args.eval_max_steps),
                "freeze_encoder_updates": int(args.freeze_encoder_updates),
                "disable_reward_shaping": bool(args.disable_reward_shaping),
                "reward_shaping": {
                    "kill_reward": kill_reward,
                    "pickup_health": pickup_health_reward,
                    "pickup_armor": pickup_armor_reward,
                    "level_clear_bonus": level_clear_bonus,
                    "ammo_spend_penalty": ammo_spend_penalty,
                    "damage_taken_penalty": damage_taken_penalty,
                    "armor_damage_penalty": armor_damage_penalty,
                },
                "bc_checkpoint": os.path.abspath(args.bc_checkpoint) if args.bc_checkpoint else None,
                "resume_from": resume_path if resume_path else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    num_updates = max(1, args.total_timesteps // args.num_steps)

    obs_buf = torch.zeros((args.num_steps, 3, args.image_size, args.image_size), device=device)
    actions_buf = torch.zeros((args.num_steps, num_actions), device=device)
    logprob_buf = torch.zeros((args.num_steps,), device=device)
    rewards_buf = torch.zeros((args.num_steps,), device=device)
    dones_buf = torch.zeros((args.num_steps,), device=device)
    values_buf = torch.zeros((args.num_steps,), device=device)

    active_scenario_idx = _scenario_index_from_step(
        global_step=global_step,
        switch_every_timesteps=args.switch_every_timesteps,
        num_scenarios=len(scenario_configs),
    )
    if active_scenario_idx != 0:
        ctrl.close()
        ctrl = DoomController(config_path=scenario_configs[active_scenario_idx])
        reward_params = _extract_reward_shaping_params(ctrl)
        kill_reward = float(reward_params["kill_reward"])
        pickup_health_reward = float(reward_params["pickup_health"])
        pickup_armor_reward = float(reward_params["pickup_armor"])
        level_clear_bonus = float(reward_params["level_clear_bonus"])
        ammo_spend_penalty = float(reward_params["ammo_spend_penalty"])
        damage_taken_penalty = float(reward_params["damage_taken_penalty"])
        armor_damage_penalty = float(reward_params["armor_damage_penalty"])
        print(f"[INFO] Escenario activo al inicio: {scenario_configs[active_scenario_idx]}")

    obs_np = ctrl.reset()
    prev_gv = gamevariables_to_dict(obs_np, gv_names)
    next_done = torch.tensor(0.0, device=device)
    episode_reward = 0.0
    episode_len = 0
    completed_ep_rewards: list[float] = []
    completed_ep_lens: list[int] = []

    eval_ctrl: Optional[DoomController] = None
    if int(args.eval_every_updates) > 0 and int(args.eval_episodes) > 0:
        eval_ctrl = DoomController(config_path=scenario_configs[active_scenario_idx], visible_window=False)
        if eval_ctrl.button_names != button_names:
            ctrl.close()
            eval_ctrl.close()
            raise RuntimeError("button_names de evaluación no coinciden con entrenamiento.")
        if eval_ctrl.game_variable_names != gv_names:
            ctrl.close()
            eval_ctrl.close()
            raise RuntimeError("game_variable_names de evaluación no coinciden con entrenamiento.")

    progress = tqdm(total=args.total_timesteps, desc="PPO Timesteps")
    progress.update(min(global_step, args.total_timesteps))

    encoder_frozen_prev: Optional[bool] = None

    try:
        for update in range(start_update, num_updates + 1):
            desired_scenario_idx = _scenario_index_from_step(
                global_step=global_step,
                switch_every_timesteps=args.switch_every_timesteps,
                num_scenarios=len(scenario_configs),
            )
            if desired_scenario_idx != active_scenario_idx:
                active_scenario_idx = desired_scenario_idx
                ctrl.close()
                ctrl = DoomController(config_path=scenario_configs[active_scenario_idx])

                reward_params = _extract_reward_shaping_params(ctrl)
                kill_reward = float(reward_params["kill_reward"])
                pickup_health_reward = float(reward_params["pickup_health"])
                pickup_armor_reward = float(reward_params["pickup_armor"])
                level_clear_bonus = float(reward_params["level_clear_bonus"])
                ammo_spend_penalty = float(reward_params["ammo_spend_penalty"])
                damage_taken_penalty = float(reward_params["damage_taken_penalty"])
                armor_damage_penalty = float(reward_params["armor_damage_penalty"])

                obs_np = ctrl.reset()
                prev_gv = gamevariables_to_dict(obs_np, gv_names)
                next_done = torch.tensor(0.0, device=device)
                episode_reward = 0.0
                episode_len = 0

                if eval_ctrl is not None:
                    eval_ctrl.close()
                    eval_ctrl = DoomController(
                        config_path=scenario_configs[active_scenario_idx],
                        visible_window=False,
                    )

                print(
                    "[INFO] Rotación de escenario "
                    f"-> idx={active_scenario_idx} cfg={scenario_configs[active_scenario_idx]} "
                    f"(global_step={global_step})"
                )

            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / float(max(1, num_updates))
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            encoder_frozen = bool(args.freeze_encoder_updates > 0 and update <= args.freeze_encoder_updates)
            if encoder_frozen_prev is None or encoder_frozen_prev != encoder_frozen:
                set_encoder_trainable(model, trainable=not encoder_frozen)
                state = "congelado" if encoder_frozen else "descongelado"
                print(f"[INFO] Encoder {state} en update {update}.")
                encoder_frozen_prev = encoder_frozen

            model.eval()
            for step in range(args.num_steps):
                global_step += 1

                obs_t = preprocess_screen(obs_np["screen"], args.image_size, device)
                obs_buf[step] = obs_t.squeeze(0)
                dones_buf[step] = next_done

                with torch.no_grad():
                    out = model.get_action_and_value(obs_t)
                action_t = out["action"].squeeze(0)

                actions_buf[step] = action_t
                logprob_buf[step] = out["log_prob"].squeeze(0)
                values_buf[step] = out["value"].squeeze(0)

                action_np = action_t.detach().cpu().numpy().astype(np.int32)
                obs_next, reward, terminated, truncated, _ = ctrl.step(action_np, repeat=max(1, int(args.repeat)))

                current_gv = gamevariables_to_dict(obs_next, gv_names)
                if use_reward_shaping:
                    shaping_extra = compute_transition_shaping(
                        prev_gv=prev_gv,
                        current_gv=current_gv,
                        kill_reward=kill_reward,
                        pickup_health_reward=pickup_health_reward,
                        pickup_armor_reward=pickup_armor_reward,
                        ammo_spend_penalty=ammo_spend_penalty,
                        damage_taken_penalty=damage_taken_penalty,
                        armor_damage_penalty=armor_damage_penalty,
                    )
                    if terminated or truncated:
                        terminal_reason_local = terminal_reason_from_step(ctrl, bool(truncated), current_gv)
                        if terminal_reason_local == "success" and level_clear_bonus != 0.0:
                            shaping_extra += float(level_clear_bonus)
                    reward = float(reward) + shaping_extra
                else:
                    reward = float(reward)

                rewards_buf[step] = float(reward)
                done = bool(terminated or truncated)
                next_done = torch.tensor(float(done), device=device)

                episode_reward += float(reward)
                episode_len += 1

                if done:
                    completed_ep_rewards.append(episode_reward)
                    completed_ep_lens.append(episode_len)
                    episode_reward = 0.0
                    episode_len = 0
                    obs_np = ctrl.reset()
                    prev_gv = gamevariables_to_dict(obs_np, gv_names)
                else:
                    obs_np = obs_next
                    if current_gv:
                        prev_gv = current_gv

            with torch.no_grad():
                next_obs_t = preprocess_screen(obs_np["screen"], args.image_size, device)
                next_value = model.get_action_and_value(next_obs_t)["value"].squeeze(0)

            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]

                delta = rewards_buf[t] + args.gamma * next_values * next_non_terminal - values_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
                advantages[t] = lastgaelam

            returns = advantages + values_buf

            b_obs = obs_buf
            b_actions = actions_buf
            b_logprob = logprob_buf
            b_advantages = advantages
            b_returns = returns
            b_values = values_buf

            b_inds = np.arange(args.num_steps)
            clipfracs = []
            approx_kls: list[float] = []

            model.train()
            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                epoch_kls: list[float] = []
                for start in range(0, args.num_steps, args.minibatch_size):
                    end = min(start + args.minibatch_size, args.num_steps)
                    mb_inds = b_inds[start:end]

                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_old_logprob = b_logprob[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]
                    mb_old_values = b_values[mb_inds]

                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    out = model.get_action_and_value(mb_obs, action=mb_actions)
                    new_logprob = out["log_prob"]
                    entropy = out["entropy"].mean()
                    new_value = out["value"]

                    logratio = new_logprob - mb_old_logprob
                    ratio = logratio.exp()
                    with torch.no_grad():
                        mb_approx_kl = ((ratio - 1.0) - logratio).mean().item()
                        approx_kls.append(float(mb_approx_kl))
                        epoch_kls.append(float(mb_approx_kl))
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(new_value - mb_old_values, -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
                if args.target_kl > 0 and epoch_kl > args.target_kl:
                    break

            approx_kl = float(np.mean(approx_kls)) if approx_kls else 0.0

            mean_reward = float(np.mean(completed_ep_rewards[-20:])) if completed_ep_rewards else 0.0
            mean_len = float(np.mean(completed_ep_lens[-20:])) if completed_ep_lens else 0.0
            clipfrac = float(np.mean(clipfracs)) if clipfracs else 0.0
            eval_mean_reward: Optional[float] = None
            eval_mean_len: Optional[float] = None
            if (
                eval_ctrl is not None
                and int(args.eval_every_updates) > 0
                and (update % int(args.eval_every_updates) == 0)
            ):
                eval_metrics = evaluate_policy(
                    model=model,
                    ctrl=eval_ctrl,
                    device=device,
                    image_size=args.image_size,
                    repeat=args.repeat,
                    num_episodes=args.eval_episodes,
                    max_steps_per_episode=args.eval_max_steps,
                    gv_names=gv_names,
                    use_reward_shaping=use_reward_shaping,
                    kill_reward=kill_reward,
                    pickup_health_reward=pickup_health_reward,
                    pickup_armor_reward=pickup_armor_reward,
                    level_clear_bonus=level_clear_bonus,
                    ammo_spend_penalty=ammo_spend_penalty,
                    damage_taken_penalty=damage_taken_penalty,
                    armor_damage_penalty=armor_damage_penalty,
                )
                eval_mean_reward = float(eval_metrics["mean_reward"])
                eval_mean_len = float(eval_metrics["mean_len"])

            print(
                f"[Update {update:04d}/{num_updates:04d}] "
                f"global_step={global_step} "
                f"scenario_idx={active_scenario_idx} "
                f"mean_ep_reward={mean_reward:.3f} "
                f"mean_ep_len={mean_len:.1f} "
                f"eval_reward={(eval_mean_reward if eval_mean_reward is not None else float('nan')):.3f} "
                f"kl={approx_kl:.5f} "
                f"clipfrac={clipfrac:.4f}"
            )

            if mean_reward > best_mean_ep_reward:
                best_mean_ep_reward = mean_reward

            selection_metric = eval_mean_reward if eval_mean_reward is not None else mean_reward

            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "button_names": button_names,
                "image_size": int(args.image_size),
                "mobilenet_model_name": str(args.mobilenet_model_name),
                "yolo_model_name": str(args.yolo_model_name),
                "yolo_imgsz": int(args.yolo_imgsz),
                "yolo_conf": float(args.yolo_conf),
                "yolo_max_det": int(args.yolo_max_det),
                "use_yolo": bool(not args.disable_yolo),
                "global_step": int(global_step),
                "update": int(update),
                "best_mean_ep_reward": float(best_mean_ep_reward),
                "best_eval_reward": float(best_eval_reward),
                "mean_ep_reward": float(mean_reward),
                "mean_ep_len": float(mean_len),
                "eval_mean_reward": None if eval_mean_reward is None else float(eval_mean_reward),
                "eval_mean_len": None if eval_mean_len is None else float(eval_mean_len),
                "selection_metric": float(selection_metric),
            }
            torch.save(state, last_path)

            if args.save_every_updates > 0 and (update % args.save_every_updates == 0):
                update_path = os.path.join(ckpt_dir, f"update_{update:04d}.pt")
                torch.save(state, update_path)
                print(f"[INFO] Checkpoint periódico guardado en: {update_path}")

            if selection_metric > best_eval_reward:
                best_eval_reward = float(selection_metric)
                state["best_eval_reward"] = float(best_eval_reward)
                state["best_mean_ep_reward"] = float(best_mean_ep_reward)
                torch.save(state, best_path)
                print(f"[INFO] Nuevo mejor PPO guardado en: {best_path}")

            progress.update(min(args.num_steps, max(0, args.total_timesteps - progress.n)))
            if global_step >= args.total_timesteps:
                break

    finally:
        progress.close()
        if eval_ctrl is not None:
            eval_ctrl.close()
        ctrl.close()

    print(f"[DONE] Entrenamiento PPO completado. Artefactos en: {run_dir}")


if __name__ == "__main__":
    main()
