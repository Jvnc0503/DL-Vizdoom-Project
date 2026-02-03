from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

import numpy as np

from vizdoom import DoomGame,Mode,Button,GameVariable,ScreenResolution,ScreenFormat


# Configuración por defecto -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "scenario": {
        "doom_scenario_path": "scenarios/basic.wad",
        "doom_map": "map01",
        "doom_skill": 3,  # 1..5
        "episode_timeout_tics": 2100,  # ~60s a 35Hz
        "episode_start_tics": 10,
        "seed": 12345,
    },
    "render": {
        "visible_window": False,
        "screen_resolution": "RES_320X240",
        "screen_format": "RGB24",
        "render_hud": False,
        "render_crosshair": False,
        "render_decals": False,
        "render_particles": False,
        "automap_buffer_enabled": False,
        "depth_buffer_enabled": False,
        "labels_buffer_enabled": False,
        "audio_enabled": False,
    },
    "controls": {
        # Orden de botones (vector MultiBinary)
        "buttons": [
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
            "ATTACK",
            "USE",
        ],
        # Variables de juego que se guardaran en obs["gamevariables"]
        "game_variables": ["HEALTH", "AMMO2", "KILLCOUNT"],
    },
    "timing": {
        "frame_skip": 4,          # tics por acción en step() si no se indica repeat
        "realtime_lock": False,   # el bloqueo de FPS suele manejarlo tu bucle externo (cuando se grabes partidas)
        "target_hz": 35,          # solo para cálculo de info["fps_estimate"]
    },
    "reward": {
        "living_reward": 0.0,
        "death_penalty": 0.0,
    },
    "recording": {
        "enable_lmp": False,         # si True, graba .lmp por episodio
        "output_dir": "recordings",  # carpeta base para episodios
        "dump_obs_every_k_tics": 1,  # 1 = cada tic; >1 para muestrear
        "enable_png_frames": False,  # si True, guarda los frames como PNGs
    },
    "safety": {
        "max_episode_seconds": 0,    # 0 = sin tope de tiempo real
        "action_timeout_ms": 0,      # 0 = sin timeout; si >0, aplica no-op si acción tarda demasiado
    },
}


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dicts without mutating the inputs."""
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("El YAML de configuración debe ser un mapeo (dict).")
    return _deep_update(DEFAULT_CONFIG, user_cfg)


def _enum_from_name(enum_cls, name: str):
    """Obtener miembro del Enum por nombre"""
    try:
        return getattr(enum_cls, name)
    except AttributeError:
        valid = [m for m in dir(enum_cls) if not m.startswith("_")]
        raise ValueError(f"Nombre no válido '{name}' para {enum_cls.__name__}. Válidos: {valid}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_bool_list(action: Union[Sequence[int], np.ndarray], length: int) -> List[int]:
    """Valida vector binario de longitud esperada y lo devuelve como lista de ints {0,1}."""
    if isinstance(action, np.ndarray):
        flat = action.reshape(-1)
        if flat.dtype.kind not in ("i", "u", "b"):  # int/uint/bool
            raise ValueError("La acción debe ser int/bool.")
        arr = flat.astype(np.int32)
    else:
        arr = np.asarray(list(action), dtype=np.int32).reshape(-1)
    if arr.size != length:
        raise ValueError(f"Acción de tamaño {arr.size}, se esperaba {length}.")
    # Convertir cualquier valor != 0 -> 1
    arr = (arr != 0).astype(np.int32)
    return arr.tolist()


@dataclass(frozen=True)
class StepResult:
    obs: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class DoomController:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.cfg = _load_yaml_config(config_path)

        # Rutas y flags de grabación
        rec_cfg = self.cfg["recording"]
        self._record_enable_lmp: bool = bool(rec_cfg.get("enable_lmp", False))
        self._record_output_dir: str = str(rec_cfg.get("output_dir", "recordings"))
        self._record_png_frames: bool = bool(rec_cfg.get("enable_png_frames", False))
        self._dump_every_k: int = int(rec_cfg.get("dump_obs_every_k_tics", 1))
        _ensure_dir(self._record_output_dir)

        # Mapeos de botones y variables de juego
        self._button_names: List[str] = list(self.cfg["controls"]["buttons"])
        self._buttons: List[Button] = [ _enum_from_name(Button, n) for n in self._button_names ]

        self._gamevar_names: List[str] = list(self.cfg["controls"]["game_variables"])
        self._gamevars: List[GameVariable] = [ _enum_from_name(GameVariable, n) for n in self._gamevar_names ]

        # Inicializa DoomGame
        self.game = DoomGame()
        self._init_game_from_config()

        # Estado de episodio
        self._episode_id: int = 0
        self._episode_dir: Optional[str] = None
        self._frames_saved: int = 0
        self._tic_counter: int = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self._start_time: float = 0.0
        self._hz: float = float(self.cfg["timing"].get("target_hz", 35))
        self._frame_skip_default: int = int(self.cfg["timing"].get("frame_skip", 4))

    # Game initialization -----------------------------------------------------------------------------
    def _init_game_from_config(self) -> None:
        scn = self.cfg["scenario"]
        rnd = self.cfg["render"]
        timing = self.cfg["timing"]
        rew = self.cfg["reward"]

        # Escenario y mapa
        self.game.set_doom_scenario_path(str(scn.get("doom_scenario_path")))
        self.game.set_doom_map(str(scn.get("doom_map", "map01")))
        self.game.set_doom_skill(int(scn.get("doom_skill", 3)))

        # Timeout y start delay
        self.game.set_episode_timeout(int(scn.get("episode_timeout_tics", 0)))
        self.game.set_episode_start_time(int(scn.get("episode_start_tics", 0)))

        # Semilla
        seed = int(scn.get("seed", 0))
        if seed != 0:
            self.game.set_seed(seed)

        # Render / buffers
        # Window visibility
        self.game.set_window_visible(bool(rnd.get("visible_window", False)))

        # Sonido (por defecto lo desactivamos)
        self.game.set_sound_enabled(bool(rnd.get("audio_enabled", False)))

        # Resolución y formato
        res_name = str(rnd.get("screen_resolution", "RES_320X240"))
        fmt_name = str(rnd.get("screen_format", "RGB24"))
        self.game.set_screen_resolution(_enum_from_name(ScreenResolution, res_name))
        self.game.set_screen_format(_enum_from_name(ScreenFormat, fmt_name))

        # Render flags
        self.game.set_render_hud(bool(rnd.get("render_hud", False)))
        self.game.set_render_crosshair(bool(rnd.get("render_crosshair", False)))
        self.game.set_render_decals(bool(rnd.get("render_decals", False)))
        self.game.set_render_particles(bool(rnd.get("render_particles", False)))

        # Buffers opcionales
        self.game.set_automap_buffer_enabled(bool(rnd.get("automap_buffer_enabled", False)))
        self.game.set_depth_buffer_enabled(bool(rnd.get("depth_buffer_enabled", False)))
        self.game.set_labels_buffer_enabled(bool(rnd.get("labels_buffer_enabled", False)))

        # Botones y variables disponibles
        self.game.set_available_buttons(self._buttons)
        self.game.set_available_game_variables(self._gamevars)

        # Recompensas simples
        self.game.set_living_reward(float(rew.get("living_reward", 0.0)))
        self.game.set_death_penalty(float(rew.get("death_penalty", 0.0)))

        # Modo: jugador sincrónico por defecto (paso a paso)
        self.game.set_mode(Mode.PLAYER)

        # Inicializar
        self.game.init()

    # API -----------------------------------------------------------------------------
    @property
    def button_names(self) -> List[str]:
        """Orden canónico de botones (acción MultiBinary)."""
        return list(self._button_names)

    @property
    def game_variable_names(self) -> List[str]:
        """Orden canónico de variables expuestas en obs['gamevariables']."""
        return list(self._gamevar_names)

    def reset(self, new_seed: Optional[int] = None) -> Dict[str, Any]:
        """Empieza un nuevo episodio y devuelve la primera observación."""
        self._episode_id += 1
        self._frames_saved = 0
        self._tic_counter = 0
        self._last_obs = None
        self._episode_dir = None
        self._start_time = time.perf_counter()

        rec_path: Optional[str] = None
        if self._record_enable_lmp:
            ep_dir = os.path.join(self._record_output_dir, f"episode_{self._episode_id:05d}")
            _ensure_dir(ep_dir)
            self._episode_dir = ep_dir
            rec_path = os.path.join(ep_dir, "episode.lmp")

        if new_seed is not None and int(new_seed) != 0:
            self.game.set_seed(int(new_seed))

        # new_episode con o sin recording_file_path
        if rec_path:
            self.game.new_episode(rec_path)
        else:
            self.game.new_episode()

        obs = self._build_obs()
        self._last_obs = obs
        if self._record_png_frames:
            self._maybe_save_frame_png(obs)
            self._save_meta_step(obs=obs, action=None, reward=0.0, term=False, trunc=False, info={"event": "reset"})
        return obs

    def step(
        self, action: Union[Sequence[int], np.ndarray], repeat: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Aplica una acción (vector binario en orden de self.button_names).
        repeat: tics a repetir (si None, usa frame_skip por defecto).
        """
        if self.game.is_episode_finished():
            # Si el usuario llama step tras terminar, devolvemos últimos obs y terminated=True
            obs = self._last_obs if self._last_obs is not None else self._build_obs(fallback_if_finished=True)
            return obs, 0.0, True, False, {"warning": "Episode already finished before step()."}

        # Validar y convertir acción
        buttons_vec = _to_bool_list(action, length=len(self._buttons))
        frame_repeat = int(repeat) if repeat is not None else self._frame_skip_default
        if frame_repeat < 1:
            frame_repeat = 1

        # Aplicar acción
        start_tic = self._tic_counter
        reward = float(self.game.make_action(buttons_vec, frame_repeat))
        self._tic_counter += frame_repeat

        # Construir observación (o fallback si terminó)
        if self.game.is_episode_finished():
            obs = self._last_obs if self._last_obs is not None else self._build_obs(fallback_if_finished=True)
            terminated = True
        else:
            obs = self._build_obs()
            self._last_obs = obs
            terminated = False

        # Reglas de truncation por tiempo real
        trunc = False
        max_secs = float(self.cfg["safety"].get("max_episode_seconds", 0))
        if max_secs > 0:
            elapsed = time.perf_counter() - self._start_time
            if elapsed >= max_secs:
                trunc = True

        # Info adicional
        elapsed = max(1e-9, time.perf_counter() - self._start_time)
        fps_est = float(self._tic_counter) / elapsed if self._hz > 0 else 0.0

        info = {
            "tics_elapsed": int(self._tic_counter),
            "tics_step": int(self._tic_counter - start_tic),
            "fps_estimate": fps_est,
        }

        # Guardado opcional de frames y metadatos
        if self._record_png_frames:
            self._maybe_save_frame_png(obs)
            self._save_meta_step(
                obs=obs, action=buttons_vec, reward=reward, term=terminated, trunc=trunc, info=info
            )

        return obs, reward, terminated, trunc, info

    def close(self) -> None:
        try:
            if self.game is not None:
                self.game.close()
        except Exception:
            pass

    # Observation builder -----------------------------------------------------------------------------
    def _build_obs(self, fallback_if_finished: bool = False) -> Dict[str, Any]:
        """Construye el dict de observación acorde a buffers activados.
        Si el episodio terminó y fallback_if_finished=True, intenta devolver el último frame válido como vacío.
        """
        state = self.game.get_state()
        obs: Dict[str, Any] = {}

        # SCREEN (siempre intentamos obtenerlo)
        if state is not None and state.screen_buffer is not None:
            screen = np.ascontiguousarray(state.screen_buffer)  # shape (C,H,W) o (H,W,C) según build

            # ViZDoom entrega por defecto (H, W, C) para RGB24
            # En caso de venir como (C,H,W), lo rotamos
            if screen.ndim == 3 and screen.shape[0] in (1, 3) and screen.shape[-1] not in (1, 3):
                # (C,H,W) -> (H,W,C)
                screen = np.transpose(screen, (1, 2, 0))
            obs["screen"] = screen.astype(np.uint8, copy=False)
            h, w = screen.shape[0], screen.shape[1]
        else:
            # Fallback si se terminó y no hay state
            if fallback_if_finished and self._last_obs is not None and "screen" in self._last_obs:
                sc = self._last_obs["screen"]
                obs["screen"] = np.zeros_like(sc)
                h, w = sc.shape[0], sc.shape[1]
            else:
                # Crear un frame mínimo si no hay nada (no debería ocurrir tras reset, pero por siacaso)
                h, w = 240, 320
                obs["screen"] = np.zeros((h, w, 3), dtype=np.uint8)

        # DEPTH
        if bool(self.cfg["render"].get("depth_buffer_enabled", False)):
            if state is not None and state.depth_buffer is not None:
                depth = np.ascontiguousarray(state.depth_buffer)
                if depth.ndim == 2:  # (H,W) -> (H,W,1)
                    depth = depth[:, :, None]
                obs["depth"] = depth
            else:
                obs["depth"] = np.zeros((h, w, 1), dtype=np.uint16)

        # LABELS (mask por píxel)
        if bool(self.cfg["render"].get("labels_buffer_enabled", False)):
            if state is not None and state.labels_buffer is not None:
                labels = np.ascontiguousarray(state.labels_buffer)
                if labels.ndim == 2:
                    labels = labels[:, :, None]
                obs["labels"] = labels.astype(np.uint8, copy=False)
            else:
                obs["labels"] = np.zeros((h, w, 1), dtype=np.uint8)

        # AUTOMAP
        if bool(self.cfg["render"].get("automap_buffer_enabled", False)):
            if state is not None and state.automap_buffer is not None:
                amap = np.ascontiguousarray(state.automap_buffer)
                if amap.ndim == 2:
                    amap = amap[:, :, None]
                obs["automap"] = amap.astype(np.uint8, copy=False)
            else:
                obs["automap"] = np.zeros((h, w, 1), dtype=np.uint8)

        # GAME VARIABLES
        if len(self._gamevars) > 0:
            if state is not None and state.game_variables is not None:
                gv = np.asarray(state.game_variables, dtype=np.float32).reshape(-1)
                obs["gamevariables"] = gv
            else:
                obs["gamevariables"] = np.zeros((len(self._gamevars),), dtype=np.float32)

        return obs

    # Recording helpers -----------------------------------------------------------------------------
    def _episode_frame_dir(self) -> Optional[str]:
        if self._episode_dir is None:
            return None
        frame_dir = os.path.join(self._episode_dir, "frames")
        _ensure_dir(frame_dir)
        return frame_dir

    def _maybe_save_frame_png(self, obs: Dict[str, Any]) -> None:
        """Guarda PNGs del screen (y opcionalmente buffers) si enable_png_frames=True."""
        if not self._record_png_frames or self._episode_dir is None:
            return
        frame_dir = self._episode_frame_dir()
        if frame_dir is None:
            return
        idx = self._frames_saved
        self._frames_saved += 1

        try:
            from imageio.v2 import imwrite 
        except Exception:
            return  # si no está imageio, no guardamos

        screen = obs.get("screen", None)
        if isinstance(screen, np.ndarray):
            imwrite(os.path.join(frame_dir, f"screen_{idx:06d}.png"), screen)

        # Buffers opcionales
        if "labels" in obs and isinstance(obs["labels"], np.ndarray):
            lab = obs["labels"]
            lab_vis = lab.squeeze()
            imwrite(os.path.join(frame_dir, f"labels_{idx:06d}.png"), lab_vis)
        if "depth" in obs and isinstance(obs["depth"], np.ndarray):
            dep = obs["depth"].squeeze()
            # Escalado simple para visualizar
            dep_vis = dep.astype(np.float32)
            if dep_vis.size > 0:
                dep_vis = dep_vis / (dep_vis.max() + 1e-9)
            dep_vis = (dep_vis * 255.0).clip(0, 255).astype(np.uint8)
            imwrite(os.path.join(frame_dir, f"depth_{idx:06d}.png"), dep_vis)

    def _save_meta_step(
        self,
        obs: Dict[str, Any],
        action: Optional[List[int]],
        reward: float,
        term: bool,
        trunc: bool,
        info: Dict[str, Any],
    ) -> None:
        """Escribe un registro JSONL """
        if self._episode_dir is None:
            return
        meta_path = os.path.join(self._episode_dir, "meta.jsonl")
        rec = {
            "tics_elapsed": int(info.get("tics_elapsed", self._tic_counter)),
            "reward": float(reward),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "action_bin": action if action is not None else None,
            "time": float(time.perf_counter() - self._start_time),
        }
        # Agregar variables útiles
        gv = obs.get("gamevariables", None)
        if isinstance(gv, np.ndarray):
            rec["gamevariables"] = gv.tolist()
            rec["gamevariable_names"] = self._gamevar_names
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def _demo_cli():
    """ Demo CLI para DoomController
    Corre un episodio haciendo random actions con la config por defecto.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Ruta a config.yaml (opcional)")
    parser.add_argument("--steps", type=int, default=300, help="Número de pasos a ejecutar")
    args = parser.parse_args()

    ctrl = DoomController(config_path=args.config)
    obs = ctrl.reset()
    print("Obs keys:", list(obs.keys()))
    print("Buttons:", ctrl.button_names)
    print("GameVariables:", ctrl.game_variable_names)

    rng = np.random.default_rng(0)
    steps = int(args.steps)
    total_r = 0.0

    for _ in range(steps):
        # Acción binaria aleatoria
        # No debería ganar nada, es un random!
        action = rng.integers(0, 2, size=len(ctrl.button_names), dtype=np.int32)
        obs, r, term, trunc, info = ctrl.step(action)
        total_r += r
        if term or trunc:
            break

    ctrl.close()

if __name__ == "__main__":
    _demo_cli()
