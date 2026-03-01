# Vizdoom — Proyecto Deep Learning (CS5364)

## Dependencias
Se recomienda usar **Python 3.12** en un entorno de **Conda**. Versiones posteriores (por ejemplo 3.13+) pueden provocar conflictos al compilar `opencv` o `vizdoom`.

## Instalación y setup

### Conda (recomendado)
Crear y actualizar el entorno desde [environment.yml](environment.yml):
```bash
conda env create -f environment.yml && conda env update -f environment.yml --prune
```
Activar el entorno:
```bash
conda activate doom
```
Eliminar el entorno:
```bash
conda deactivate
conda env remove --name doom
```

> Nota: [environment.yml](environment.yml) usa `conda-forge` como canal principal y especifica `python=3.12`.

### Virtualenv (no recomendado)
Se puede usar un `venv`, pero puede haber incompatibilidades de paquetes:
```bash
python -m venv .venv
source .venv/bin/activate
```
Desactivar y eliminar:
```bash
deactivate
rm -rf .venv/
```

## Mapeo de controles
Configura las teclas en [keymap.yaml](keymap.yaml). Ejemplo de mapeo:
```yaml
W: MOVE_FORWARD
S: MOVE_BACKWARD
A: MOVE_LEFT
D: MOVE_RIGHT
LEFT: TURN_LEFT
RIGHT: TURN_RIGHT
SPACE: ATTACK
E: USE
SHIFT: SPEED
"1": SELECT_WEAPON1
"2": SELECT_WEAPON2
"3": SELECT_WEAPON3
"4": SELECT_WEAPON4
"5": SELECT_WEAPON5
"6": SELECT_WEAPON6
"7": SELECT_WEAPON7
ESCAPE: QUIT
```

## Ejecución
Modo de juego (normal):
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml
```

Grabar partida:
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml --record
```

## Selección de WAD, mapa y dificultad
La selección se hace en `game_config.yaml`, dentro de la sección `scenario`:

```yaml
scenario:
  doom_scenario_path: "scenarios/doom.wad"
  doom_map: "e3m2"
  doom_skill: 3
```

### 1) Elegir WAD (`doom_scenario_path`)
- Define el archivo WAD que se cargará.
- Ejemplos comunes en este repo: `scenarios/doom.wad`, `scenarios/doom2.wad`, `scenarios/plutonia.wad`, `scenarios/tnt.wad`.

Ejemplo:
```yaml
scenario:
  doom_scenario_path: "scenarios/doom2.wad"
```

### 2) Elegir mapa (`doom_map`)
- En DOOM clásico se usan nombres tipo `E#M#` (por ejemplo `e1m1`, `e3m2`).
- En DOOM II y algunos WADs modernos se usan nombres tipo `MAP##` (por ejemplo `map01`, `map07`).
- Si un mapa no existe en el WAD seleccionado, la partida puede fallar al iniciar.

Ejemplos:
```yaml
scenario:
  doom_scenario_path: "scenarios/doom.wad"
  doom_map: "e1m1"
```

```yaml
scenario:
  doom_scenario_path: "scenarios/doom2.wad"
  doom_map: "map01"
```

### 3) Elegir dificultad (`doom_skill`)
`doom_skill` va de `1` a `5`:
- `1`: I'm Too Young To Die (muy fácil)
- `2`: Hey, Not Too Rough (fácil)
- `3`: Hurt Me Plenty (normal)
- `4`: Ultra-Violence (difícil)
- `5`: Nightmare! (muy difícil)

Ejemplo (subir dificultad):
```yaml
scenario:
  doom_skill: 4
```

### Configuración sugerida para empezar
```yaml
scenario:
  doom_scenario_path: "scenarios/doom.wad"
  doom_map: "e1m1"
  doom_skill: 2
```

Después de cambiar estos valores, ejecuta normalmente:
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml
```

## Recompensas
Puedes ajustar pesos de recompensa en [game_config.yaml](game_config.yaml). Ejemplo:
```yaml
reward:
  living_reward: -0.02
  death_penalty: -500
```

## Notas adicionales
- Si tienes problemas instalando `vizdoom` o `opencv`, revisa que tu toolchain (compiladores, headers) esté instalado y que el entorno use `python 3.12`.
- Para reproducibilidad, usa siempre `conda` y el archivo [environment.yml](environment.yml) incluido en este repositorio.

## Aviso sobre assets, WADs y licencia
- La carpeta [scenarios](scenarios) puede contener configuraciones y recursos de ejemplo de ViZDoom que son abiertos o libres de uso y redistribución para pruebas/entrenamiento.
- Los IWADs/recursos comerciales (por ejemplo `doom.wad`, `doom2.wad`, `plutonia.wad`, `tnt.wad`, etc.) **no forman parte de la distribución del repositorio** y se omiten mediante reglas de ignore (por ejemplo en [.gitignore](.gitignore)).
- Los archivos usados localmente para pruebas en este proyecto fueron extraídos desde una copia propia del juego adquirida legalmente.
- Cualquier persona que clone este repositorio debe proveer sus propios assets/WADs con una licencia válida antes de ejecutar escenarios que los requieran.