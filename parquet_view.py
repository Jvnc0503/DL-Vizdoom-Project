import pandas as pd
from pathlib import Path
import sys

# Usar la carpeta `recordings` relativa al directorio de trabajo del proyecto
recordings_path = Path.cwd() / "recordings"

if not recordings_path.exists():
    print(f"No se encontró la carpeta de grabaciones: {recordings_path}")
    print("Asegúrate de ejecutar el script desde el directorio del proyecto o crea la carpeta recordings.")
    sys.exit(1)

folders = [f for f in recordings_path.iterdir() if f.is_dir()]

for folder in folders:
    parquet_files = list(folder.glob("*.parquet"))
    if not parquet_files:
        continue

    for file in parquet_files:
        df = pd.read_parquet(file)
        print(f"Archivo: {file.name}, Filas: {len(df)}")
        break
    break