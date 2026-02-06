"""
doom_play.py
- Modo libre (por defecto): prueba escenarios y practica sin grabar.
- Modo grabación (--record): guarda partida (procura guardar luego de dominar el escenario, para que el modelo aprenda buenas estrategias de juego).

Uso:
  python doom_play.py --config game_config.yaml --keymap keymap.yaml
  python doom_play.py --config game_config.yaml --keymap keymap.yaml --record
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import threading
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Set, Optional, Any

import numpy as np
import pandas as pd

import yaml
from imageio.v2 import imwrite
from pynput import keyboard
import cv2

from doom_controller import DoomController


# Defaults de grabación -----------------------------------------------------------------------------
VIDEO_FPS_DEFAULT = 35.0
VIDEO_BACKEND_DEFAULT = "ffmpeg"      # "ffmpeg" | "opencv" | "npz"
VIDEO_CONTAINER_DEFAULT = "mkv"       # "mkv" | "mp4"
VIDEO_CODEC_DEFAULT = "libx264"       # ffmpeg: "libx264" | "libx265" | "ffv1" | "prores_ks" | "libvpx-vp9"
VIDEO_CRF_DEFAULT = 18                # 0 = lossless, 18 se ve bien
VIDEO_PRESET_DEFAULT = "medium"     # x264: ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow/placebo
CHUNK_SIZE_DEFAULT = 350
QUEUE_MAXSIZE_DEFAULT = 256

# Utilidades ----------------------------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encuentra el archivo YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un mapeo (dict): {path}")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Teclado ------------------------------------------------------------------------------------------

KEY_ALIASES = {
    "ctrl_l": "CTRL",
    "ctrl_r": "CTRL",
    "ctrl": "CTRL",
    "shift_l": "SHIFT",
    "shift_r": "SHIFT",
    "shift": "SHIFT",
    "space": "SPACE",
    "esc": "ESCAPE",
    "enter": "ENTER",
    "return": "ENTER",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
}

def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
    try:
        if isinstance(key, keyboard.KeyCode):
            ch = key.char
            if ch is None:
                return None
            if len(ch) == 1:
                if ch.isalpha():
                    return ch.upper()
                else:
                    return ch
            return ch.upper()
        else:
            name = str(key).split('.')[-1].lower()
            return KEY_ALIASES.get(name, name.upper())
    except Exception:
        return None


class GlobalKeyState:
    def __init__(self) -> None:
        self._pressed: Set[str] = set()
        self._lock = threading.Lock()
        self._exit_requested = False
        self._listener: Optional[keyboard.Listener] = None

    def on_press(self, key) -> None:
        k = normalize_key(key)
        if k is None:
            return
        with self._lock:
            self._pressed.add(k)
            if k == "ESCAPE":
                self._exit_requested = True

    def on_release(self, key) -> None:
        k = normalize_key(key)
        if k is None:
            return
        with self._lock:
            self._pressed.discard(k)

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def snapshot(self) -> Set[str]:
        with self._lock:
            return set(self._pressed)

    def exit_requested(self) -> bool:
        with self._lock:
            return bool(self._exit_requested)

    def clear_exit(self) -> None:
        with self._lock:
            self._exit_requested = False


# Mapeo de teclas a botones ------------------------------------------------------------------------

def build_action_vector(button_names: List[str], keymap: Dict[str, Any], pressed: Set[str]) -> np.ndarray:
    K = len(button_names)
    vec = np.zeros((K,), dtype=np.int32)
    keyboard_map: Dict[str, str] = {k.upper(): v for k, v in keymap.get("keyboard", {}).items()}
    for key in pressed:
        btn = keyboard_map.get(key)
        if btn is None or btn == "QUIT":
            continue
        try:
            j = button_names.index(btn)
            vec[j] = 1
        except ValueError:
            pass
    return vec


# Grabación ----------------------------------------------------------------------------------------

def _open_ffmpeg_stdin(w: int, h: int, fps: float, codec: str, preset: str, crf: int, out_path: str) -> Any:
    """
    Abre un proceso ffmpeg que toma rawvideo bgr24 por stdin y escribe al contenedor.
    Devuelve Popen con stdin listo para write().
    """
    import subprocess

    args = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(float(fps)),
        "-i", "-",          # stdin
        "-an",
        "-threads", "0",
    ]

    # Selección de codec y parámetros
    if codec in ("libx264", "libx265"):
        args += ["-c:v", codec, "-preset", preset, "-crf", str(int(crf))]
        if codec == "libx265" and int(crf) == 0:
            args += ["-x265-params", "lossless=1"]
    elif codec == "ffv1":
        args += ["-c:v", "ffv1", "-level", "3", "-g", "1"]
    elif codec == "prores_ks":
        args += ["-c:v", "prores_ks", "-profile:v", "3"]
    elif codec == "libvpx-vp9":
        args += ["-c:v", "libvpx-vp9", "-lossless", "1", "-row-mt", "1"]
    else:
        args += ["-c:v", codec]

    args += [out_path]

    try:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError("No se encontró 'ffmpeg' en el PATH. Instálalo o ajusta recording.video_backend.") from e
    except Exception as e:
        raise RuntimeError(f"No se pudo iniciar ffmpeg con args: {args}") from e

    if proc.stdin is None:
        raise RuntimeError("ffmpeg no expuso stdin; no se podrá enviar video.")
    return proc

def _writer_process(
    queue: mp.Queue,
    session_dir: str,
    button_names: List[str],
    gamevar_names: List[str],
    doom_wad: str,
    doom_map: str,
    doom_skill: int,
    
    video_backend: str,
    video_container: str,
    video_codec: str,
    video_crf: int,
    video_preset: str,
    video_fps: float,
    chunk_size: int,
) -> None:
    """
    Recibe:
      - 'frame': {screen, depth, labels, automap, rec_row}
      - 'finalize': {terminal_reason, extra_meta}
    Guarda:
      - screen.avi (MJPG/XVID/etc. según 'video_codec')
      - depth_chunk_XXX.npz, labels_chunk_XXX.npz, automap_chunk_XXX.npz
      - meta.parquet, session_meta.json
    """
    import traceback

    frames_dir = os.path.join(session_dir, "frames")
    ensure_dir(frames_dir)

    # Recursos de video (uno u otro backend)
    vw = None               # cv2.VideoWriter, si backend == opencv
    ff_proc = None          # subprocess Popen, si backend == ffmpeg
    video_path = os.path.join(
        session_dir,
        f"screen.{video_container if video_backend=='ffmpeg' else 'avi'}"
    )

    if video_backend == "opencv":
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if video_codec.upper() == "XVID" else "MJPG"))
    # Buffers de chunks
    depth_buf: List[np.ndarray] = []
    labels_buf: List[np.ndarray] = []
    automap_buf: List[np.ndarray] = []
    screen_buf: List[np.ndarray] = [] 
    chunk_idx = 0

    # Meta
    records: List[Dict[str, Any]] = []
    cumulative_reward: float = 0.0

    def flush_chunk() -> None:
        nonlocal chunk_idx, depth_buf, labels_buf, automap_buf, screen_buf
        if len(depth_buf) > 0:
            np.savez_compressed(
                os.path.join(session_dir, f"depth_chunk_{chunk_idx:03d}.npz"),
                frames=np.stack(depth_buf, axis=0),
            )
            depth_buf = []
        if len(labels_buf) > 0:
            np.savez_compressed(
                os.path.join(session_dir, f"labels_chunk_{chunk_idx:03d}.npz"),
                frames=np.stack(labels_buf, axis=0),
            )
            labels_buf = []
        if len(automap_buf) > 0:
            np.savez_compressed(
                os.path.join(session_dir, f"automap_chunk_{chunk_idx:03d}.npz"),
                frames=np.stack(automap_buf, axis=0),
            )
            automap_buf = []
        if video_backend == "npz" and len(screen_buf) > 0:
            np.savez_compressed(
                os.path.join(session_dir, f"screen_chunk_{chunk_idx:03d}.npz"),
                frames=np.stack(screen_buf, axis=0),
            )
            screen_buf = []
        chunk_idx += 1

    try:
        while True:
            item = queue.get()
            if item is None:
                break

            itype = item.get("type", None)

            if itype == "frame":
                screen = item["screen"]              # (H,W,3) uint8
                depth = item.get("depth", None)      # (H,W) uint16 o None
                labels = item.get("labels", None)    # (H,W) uint8  o None
                automap = item.get("automap", None)  # (H,W) o (H,W,3) uint8 o None
                rec_row = item["rec_row"]

                # Inicializar backend de video con la primera frame
                if video_backend == "ffmpeg":
                    if ff_proc is None:
                        h, w = int(screen.shape[0]), int(screen.shape[1])
                        out_path = os.path.join(session_dir, f"screen.{video_container}")
                        ff_proc = _open_ffmpeg_stdin(
                            w=w, h=h, fps=video_fps, codec=video_codec,
                            preset=video_preset, crf=int(video_crf), out_path=out_path
                        )
                    # OpenCV usa BGR; convertimos RGB->BGR
                    bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    ff_proc.stdin.write(bgr.tobytes())

                elif video_backend == "opencv":
                    if vw is None:
                        h, w = int(screen.shape[0]), int(screen.shape[1])
                        vw = cv2.VideoWriter(video_path, fourcc, float(video_fps), (w, h), True)
                        if (not vw) or (not vw.isOpened()):
                            raise RuntimeError("No se pudo abrir el VideoWriter de OpenCV.")
                    bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    vw.write(bgr)

                elif video_backend == "npz":
                    screen_buf.append(screen.astype(np.uint8, copy=False))

                # Acumular depth/labels/automap
                if isinstance(depth, np.ndarray):
                    if depth.ndim == 3 and depth.shape[2] == 1:
                        depth = depth[:, :, 0]
                    depth_buf.append(depth.astype(np.uint16, copy=False))
                if isinstance(labels, np.ndarray):
                    if labels.ndim == 3 and labels.shape[2] == 1:
                        labels = labels[:, :, 0]
                    labels_buf.append(labels.astype(np.uint8, copy=False))
                if isinstance(automap, np.ndarray):
                    automap_buf.append(automap.astype(np.uint8, copy=False))

                # Volcado por tamaño de chunk
                max_len = max(len(depth_buf), len(labels_buf), len(automap_buf), len(screen_buf))
                if max_len >= int(chunk_size):
                    flush_chunk()

                # Meta por frame
                cumulative_reward += float(rec_row.get("reward", 0.0))
                rec_row["cumulative_reward_video"] = cumulative_reward
                records.append(rec_row)

            elif itype == "finalize":
                # Descargar buffers restantes
                if (len(depth_buf) > 0) or (len(labels_buf) > 0) or (len(automap_buf) > 0) or (len(screen_buf) > 0):
                    flush_chunk()

                # Escribir meta
                df = pd.DataFrame(records)
                df.to_parquet(os.path.join(session_dir, "meta.parquet"), index=False)

                meta = {
                    "button_names": item["button_names"],
                    "gamevariable_names": item["gamevariable_names"],
                    "num_steps": len(records),
                    "terminal_reason": item["terminal_reason"],
                    "cumulative_reward": cumulative_reward,
                    "doom_wad": doom_wad,
                    "doom_map": doom_map,
                    "doom_skill": int(doom_skill),
                    "video_backend": video_backend,
                    "video_path": (
                        f"screen.{video_container}" if video_backend == "ffmpeg"
                        else ("screen.avi" if video_backend == "opencv" else "screen_chunk_XXX.npz")
                    ),
                    "video_fps": float(video_fps),
                    "video_codec": str(video_codec),
                    "video_crf": int(video_crf),
                    "video_preset": str(video_preset),
                    "chunk_size": int(chunk_size),
                }
                extra_meta = item.get("extra_meta", None)
                if isinstance(extra_meta, dict):
                    meta.update(extra_meta)
                with open(os.path.join(session_dir, "session_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                # Cerrar recursos de video
                if ff_proc is not None:
                    try:
                        ff_proc.stdin.close()
                        ff_proc.wait()
                    except Exception:
                        pass
                if vw is not None:
                    try:
                        vw.release()
                    except Exception:
                        pass
                break

    except Exception as e:
        # Liberar recursos en caso de error
        try:
            if ff_proc is not None:
                ff_proc.stdin.close()
                ff_proc.wait(timeout=2)
        except Exception:
            pass
        try:
            if vw is not None:
                vw.release()
        except Exception:
            pass
        import traceback
        with open(os.path.join(session_dir, "writer_error.log"), "w", encoding="utf-8") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())



class AsyncEpisodeRecorder:
    """
    Guarda:
      - frames (screen/depth/labels)
      - meta.parquet (action/rewards/flags/vars + doom_* info)
      - session_meta.json (resumen de la sesión)
    """
    def __init__(
        self,
        base_dir: str,
        button_names: List[str],
        gamevar_names: List[str],
        doom_wad: str,
        doom_map: str,
        doom_skill: int,
        
        video_backend: str,
        video_container: str,
        video_codec: str,
        video_crf: int,
        video_preset: str,
        video_fps: float,
        chunk_size: int,
        queue_maxsize: int,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{ts}")
        ensure_dir(self.session_dir)

        self.button_names = list(button_names)
        self.gamevar_names = list(gamevar_names)

        self.queue: mp.Queue = mp.Queue(maxsize=int(queue_maxsize))
        self.proc = mp.Process(
            target=_writer_process,
            args=(
                self.queue,
                self.session_dir,
                self.button_names,
                self.gamevar_names,
                doom_wad,
                doom_map,
                doom_skill,
                video_backend,
                video_container,
                video_codec,
                int(video_crf),
                video_preset,
                float(video_fps),
                int(chunk_size),
            ),
            daemon=True,
        )
        self.proc.start()

    def enqueue_step(
        self,
        t_index: int,
        obs: Dict[str, Any],
        action_bin: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        lives: int,
        reason: Optional[str],
        timestamp_s: Optional[float],
        doom_wad: str,
        doom_map: str,
        doom_skill: int,
    ) -> None:
        gv_dict: Dict[str, float] = {}
        if "gamevariables" in obs and isinstance(obs["gamevariables"], np.ndarray):
            gvs = obs["gamevariables"].reshape(-1)
            for name, val in zip(self.gamevar_names, gvs.tolist()):
                gv_dict[name] = float(val)

        health = gv_dict.get("HEALTH", np.nan)
        armor = gv_dict.get("ARMOR", np.nan)
        killcount = gv_dict.get("KILLCOUNT", np.nan)

        selected_weapon = gv_dict.get("SELECTED_WEAPON", np.nan)
        selected_weapon_ammo = gv_dict.get("SELECTED_WEAPON_AMMO", np.nan)
        ammo1 = gv_dict.get("AMMO1", np.nan)
        ammo2 = gv_dict.get("AMMO2", np.nan)
        ammo3 = gv_dict.get("AMMO3", np.nan)
        ammo4 = gv_dict.get("AMMO4", np.nan)
        weapon1 = gv_dict.get("WEAPON1", np.nan)
        weapon2 = gv_dict.get("WEAPON2", np.nan)
        weapon3 = gv_dict.get("WEAPON3", np.nan)
        weapon4 = gv_dict.get("WEAPON4", np.nan)
        weapon5 = gv_dict.get("WEAPON5", np.nan)
        weapon6 = gv_dict.get("WEAPON6", np.nan)
        weapon7 = gv_dict.get("WEAPON7", np.nan)

        rec_row = {
            "t_index": int(t_index),
            "action_bin": action_bin.astype(np.int8).tolist(),
            "action_names": self.button_names,
            "reward": float(reward),
            "is_terminal": bool(terminated),
            "is_timeout": bool(truncated),
            "lives": int(lives),
            "health": health,
            "armor": armor,
            "killcount": killcount,
            "timestamp_s": float(timestamp_s) if timestamp_s is not None else None,
            "terminal_reason": reason,
            "doom_wad": doom_wad,
            "doom_map": doom_map,
            "doom_skill": int(doom_skill),
            "selected_weapon": selected_weapon,
            "selected_weapon_ammo": selected_weapon_ammo,
            "ammo1": ammo1,
            "ammo2": ammo2,
            "ammo3": ammo3,
            "ammo4": ammo4,
            "weapon1": weapon1,
            "weapon2": weapon2,
            "weapon3": weapon3,
            "weapon4": weapon4,
            "weapon5": weapon5,
            "weapon6": weapon6,
            "weapon7": weapon7,
        }

        screen = obs.get("screen", None)
        depth = obs.get("depth", None)
        labels = obs.get("labels", None)
        automap = obs.get("automap", None)

        if isinstance(depth, np.ndarray) and depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if isinstance(labels, np.ndarray) and labels.ndim == 3 and labels.shape[2] == 1:
            labels = labels[:, :, 0]

        self.queue.put({
            "type": "frame",
            "screen": screen,
            "depth": depth,
            "labels": labels,
            "automap": automap,
            "rec_row": rec_row,
        })

    def finalize(self, terminal_reason: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        self.queue.put({
            "type": "finalize",
            "terminal_reason": terminal_reason,
            "button_names": self.button_names,
            "gamevariable_names": self.gamevar_names,
            "extra_meta": extra_meta or {},
        })
        self.proc.join(timeout=60.0)



# Loop principal del juego ----------------------------------------------------------------------------------------

def ask_to_start() -> bool:
    try:
        ans = input("¿Deseas comenzar? [s/N]: ").strip().lower()
        return ans in ("s", "si", "sí", "y", "yes")
    except KeyboardInterrupt:
        return False


def main():
    parser = argparse.ArgumentParser(description="DOOM")
    parser.add_argument("--config", type=str, default="game_config.yaml", help="Ruta al config YAML.")
    parser.add_argument("--keymap", type=str, default="keymap.yaml", help="Ruta al mapeo de teclas.")
    parser.add_argument("--record", action="store_true", help="Si se indica, habilita modo grabación.")
    parser.add_argument("--output-dir", type=str, default="recordings", help="Directorio base para sesiones grabadas.")
    args = parser.parse_args()

    print("\n=== DOOM ===")
    if not ask_to_start():
        print("Saliendo.")
        return

    # Cargar configuraciones
    try:
        game_cfg = load_yaml(args.config)
    except Exception as e:
        print(f"ERROR al cargar config de juego: {e}")
        sys.exit(1)

    try:
        keymap = load_yaml(args.keymap)
    except Exception as e:
        print(f"ERROR al cargar keymap: {e}")
        sys.exit(1)

    # Extraer info (wad/map/skill)
    scn = game_cfg.get("scenario", {})
    doom_wad = str(scn.get("doom_scenario_path", "unknown_wad"))
    doom_map = str(scn.get("doom_map", "map01"))
    doom_skill = int(scn.get("doom_skill", 3))

    rec_cfg = game_cfg.get("recording", {}) or {}
    video_backend = str(rec_cfg.get("video_backend", VIDEO_BACKEND_DEFAULT)).lower()
    video_container = str(rec_cfg.get("video_container", VIDEO_CONTAINER_DEFAULT))
    video_codec = str(rec_cfg.get("video_codec", VIDEO_CODEC_DEFAULT))
    video_crf = int(rec_cfg.get("video_crf", VIDEO_CRF_DEFAULT))
    video_preset = str(rec_cfg.get("video_preset", VIDEO_PRESET_DEFAULT))
    video_fps = float(rec_cfg.get("video_fps", VIDEO_FPS_DEFAULT))
    chunk_size = int(rec_cfg.get("chunk_size", CHUNK_SIZE_DEFAULT))
    queue_maxsize = int(rec_cfg.get("queue_maxsize", QUEUE_MAXSIZE_DEFAULT))

    # Iniciar controlador
    try:
        ctrl = DoomController(config_path=args.config)
    except Exception as e:
        print(f"ERROR al crear DoomController con config '{args.config}':\n{e}")
        sys.exit(1)

    # Captura de teclado
    keys = GlobalKeyState()
    keys.start()
    print("Controles activos. Pulsa ESC para salir.")

    # Reset y metadatos de espacios
    obs = ctrl.reset()
    button_names = ctrl.button_names
    gv_names = ctrl.game_variable_names

    # Reward shaping params (desde game_config.yaml)
    rew_cfg = game_cfg.get("reward", {}) if isinstance(game_cfg, dict) else {}
    kill_reward = float(rew_cfg.get("kill_reward", 0.0))
    pickup_health_reward = float(rew_cfg.get("pickup_health", 0.0))
    pickup_armor_reward = float(rew_cfg.get("pickup_armor", 0.0))
    level_clear_bonus = float(rew_cfg.get("level_clear_bonus", 0.0))

    # Estado anterior de gamevariables para computar deltas
    prev_gv: Optional[Dict[str, float]] = None

    # Recorder (si esta activado)
    recorder: Optional[AsyncEpisodeRecorder] = None
    if args.record:
        ensure_dir(args.output_dir)
        recorder = AsyncEpisodeRecorder(
            base_dir=args.output_dir,
            button_names=button_names,
            gamevar_names=gv_names,
            doom_wad=doom_wad,
            doom_map=doom_map,
            doom_skill=doom_skill,
            video_backend=video_backend,
            video_container=video_container,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
            video_fps=video_fps,
            chunk_size=chunk_size,
            queue_maxsize=queue_maxsize,
        )
        print(f"Grabación activa. La sesión se guardará en: {recorder.session_dir}\n")

    # Bucle a 35 Hz
    target_hz = 35.0
    period = 1.0 / target_hz
    next_t = time.perf_counter()
    t_index = 0
    cumulative_reward = 0.0
    lives = 1

    terminal_reason: Optional[str] = None

    try:
        while True:
            if keys.exit_requested():
                terminal_reason = "user"
                break

            pressed = keys.snapshot()
            action_vec = build_action_vector(button_names, keymap, pressed)

            # 1 tic por iteración -> 35 Hz reales
            obs, r, terminated, truncated, info = ctrl.step(action_vec, repeat=1)

            # Extraer gamevariables y calcular shaping (kills / pickups)
            shaping_extra = 0.0
            gv = obs.get("gamevariables", None)
            current_gv: Dict[str, float] = {}
            if isinstance(gv, np.ndarray):
                vals = gv.reshape(-1).tolist()
                for name, val in zip(gv_names, vals):
                    current_gv[name] = float(val)

            if prev_gv is not None and current_gv:
                # Kills
                if "KILLCOUNT" in current_gv and "KILLCOUNT" in prev_gv:
                    delta_k = int(current_gv.get("KILLCOUNT", 0) - prev_gv.get("KILLCOUNT", 0))
                    if delta_k > 0:
                        shaping_extra += float(delta_k) * kill_reward
                # Health pickups (only positive deltas)
                if "HEALTH" in current_gv and "HEALTH" in prev_gv:
                    delta_h = float(current_gv.get("HEALTH", 0.0) - prev_gv.get("HEALTH", 0.0))
                    if delta_h > 0:
                        # Normalizamos por 25 HP para que pickups pequeños no exploten
                        shaping_extra += (delta_h / 25.0) * pickup_health_reward
                # Armor pickups
                if "ARMOR" in current_gv and "ARMOR" in prev_gv:
                    delta_a = float(current_gv.get("ARMOR", 0.0) - prev_gv.get("ARMOR", 0.0))
                    if delta_a > 0:
                        shaping_extra += (delta_a / 25.0) * pickup_armor_reward

            # Aplicar shaping al reward del paso
            r = float(r) + shaping_extra

            # Determinar motivo terminal localmente (para aplicar bonificación de nivel)
            terminal_reason_local: Optional[str] = None
            if terminated or truncated:
                if truncated:
                    terminal_reason_local = "timeout"
                else:
                    try:
                        is_dead = bool(ctrl.game.is_player_dead())
                    except Exception:
                        is_dead = False
                    if is_dead:
                        terminal_reason_local = "death"
                    else:
                        health_val = None
                        if isinstance(gv, np.ndarray) and "HEALTH" in gv_names:
                            health_val = float(gv[gv_names.index("HEALTH")])
                        terminal_reason_local = "death" if (health_val is not None and health_val <= 0) else "success"
                # Si completó el nivel, añadir bonificación de final de nivel
                if terminal_reason_local == "success" and level_clear_bonus != 0.0:
                    r += float(level_clear_bonus)

            # Registrar paso (incluyendo el paso final si terminó/truncó)
            if recorder is not None:
                recorder.enqueue_step(
                    t_index=t_index,
                    obs=obs,
                    action_bin=action_vec,
                    reward=r,
                    terminated=terminated,
                    truncated=truncated,
                    lives=lives,
                    reason=terminal_reason_local,
                    timestamp_s=(time.perf_counter() - next_t + period),
                    doom_wad=doom_wad,
                    doom_map=doom_map,
                    doom_skill=doom_skill,
                )

            # Acumular recompensa (incluyendo shaping)
            cumulative_reward += float(r)

            # Actualizar prev_gv
            if current_gv:
                prev_gv = current_gv

            if terminated or truncated:
                terminal_reason = terminal_reason_local
                break

            t_index += 1

            # Sincronizar a 35 Hz
            next_t += period
            now = time.perf_counter()
            delay = next_t - now
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = now

    except KeyboardInterrupt:
        terminal_reason = "user"
    finally:
        if recorder is not None:
            recorder.finalize(
                terminal_reason=terminal_reason or "user",
                extra_meta={
                    "total_steps": t_index,
                    "buttons": button_names,
                    "gamevariables": gv_names,
                    "config_path": os.path.abspath(args.config),
                    "keymap_path": os.path.abspath(args.keymap),
                    "cumulative_reward": cumulative_reward,
                    "dataset_format": {
                        "video": f"screen.{video_container} ({video_codec}, crf={video_crf}, preset={video_preset}, {video_fps} fps)",
                        "chunks": {
                            "depth": "depth_chunk_XXX.npz (uint16, (N,H,W))",
                            "labels": "labels_chunk_XXX.npz (uint8, (N,H,W))",
                            "automap": "automap_chunk_XXX.npz (uint8, (N,H,W) o (N,H,W,3))",
                            "chunk_size": chunk_size
                        },
                        "meta": "meta.parquet (una fila por frame de video)",
                    },
                },
            )

        msg = {
            "user": "Sesión terminada por el usuario.",
            "death": "Perdiste! :c (muerte del jugador).",
            "success": "Ganaste! :D (Objetivo alcanzado).",
            "timeout": "Sesión terminada por límite de tiempo.",
            None: "Sesión finalizada por X motivo.",
        }.get(terminal_reason, "Sesión finalizada.")
        print("\n" + "=" * 60)
        print(f"Motivo de cierre: {terminal_reason}")
        print(msg)
        print("=" * 60 + "\n")
        time.sleep(3.0)

        keys.stop()
        ctrl.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
