# Vizdoom Project - Deep Learning (CS5364)

## Dependencias
Se recomienda encarecidamente usar **Python 3.12** con un environment de **Conda**, versiones posteriores (como 3.13 o 3.14) producen conflictos de librerías y errores al compilar las librerías opencv y/o vizdoom.

## Setup

### Conda
Creación y activación:
```bash
conda env create -f environment.yml && conda env update -f environment.yml --prune
```
```bash
conda activate doom
```
Borrar env:
```bash
conda deactivate
```
```bash
conda env remove --name doom
```

### Venv
**ADVERTENCIA**: Se desaconseja usar environments regulares porque no asegura compatibilidad entre versiones de dependencias.
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
Desactivar:
```bash
deactivate
```
```bash
rm -rf .venv/
```

## Control Mapping
Se puede configurar la asignación de botones en el archivo [keymap.yaml](keymap.yaml)
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
Modo de juego normal:
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml
```
Modo de grabación:
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml --record
```

## Rewards
Se puede cambiar el peso de las recompensas en [game_config.yaml](game_config.yaml)
```yaml
reward:
  living_reward: -0.02
  death_penalty: -500
```