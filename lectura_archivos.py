import os, glob, time, random
import pandas as pd
from joblib import Parallel, delayed
from ModeloOptimizado import calcause
from tqdm import tqdm
from joblib import parallel_backend
import numpy as np

random.seed(0)
INPUT_COVS = "input_2016.csv"
BASE_PATH = "data/data_cf_all"
CARPETAS = range(1, 78)

# Carpetas auxiliares para guardar resultados de lotes y errores
os.makedirs("resultados_lotes", exist_ok=True)
os.makedirs("resultados_done", exist_ok=True)

def procesar_archivo(archivo, n_boot=500):
    """
    Procesa un archivo CSV, calcula el SATT y guarda los resultados.
    """
    archivo_done = os.path.join("resultados_done", os.path.basename(archivo) + ".done")
    if os.path.exists(archivo_done):
        return None

    try:
        inicio = time.time()
        zymu = pd.read_csv(archivo).reset_index(drop=True)
        x    = pd.read_csv(INPUT_COVS).reset_index(drop=True)

        zymu['y'] = zymu['z'] * zymu['y1'] + (1 - zymu['z']) * zymu['y0']
        data = pd.concat([x, zymu[['z', 'y']]], axis=1)
        covs = [c for c in data.select_dtypes('number') if c.startswith("x_")]

        satt_true = (zymu.mu1 - zymu.mu0)[zymu.z == 1].mean()
        res = calcause(data, covs, n_bootstrap=n_boot, random_state=42)

        bias  = res['satt'] - satt_true
        mse   = bias**2
        cover = res['ci_95'][0] <= satt_true <= res['ci_95'][1]
        ancho = res['ci_ancho']
        dur   = time.time() - inicio

        with open(archivo_done, 'w') as f:
            f.write("ok")

        return {
            'carpeta'       : os.path.basename(os.path.dirname(archivo)),
            'archivo'       : os.path.basename(archivo),
            'satt_true'     : satt_true,
            'satt_est'      : res['satt'],
            'bias'          : bias,
            'mse'           : mse,
            'ci_95'         : str(res['ci_95']),
            'ci_ancho'      : ancho,
            'cobertura_ic'  : cover,
            'modelo_elegido': res['modelo_elegido'],
            'params_rf'     : str(res['params_rf']),
            'params_gp'     : str(res['params_gp']),
            'tiempo_seg'    : dur
        }

    except Exception as e:
        return {'archivo': os.path.basename(archivo), 'error': str(e)}

# Procesamiento de archivos en paralelo con manejo de errores
for i in CARPETAS:
    carpeta = os.path.join(BASE_PATH, str(i))
    archivos = glob.glob(os.path.join(carpeta, "zymu_*.csv"))
    seleccionados = random.sample(archivos, min(20, len(archivos)))

    with parallel_backend('loky', n_jobs=-1):
        resultados = Parallel()(
            delayed(procesar_archivo)(f) for f in tqdm(seleccionados, desc=f"Procesando carpeta {i:02}")
        )

    resultados = [r for r in resultados if r is not None]
    if resultados:
        df = pd.DataFrame(resultados)
        df.to_csv(f"resultados_lotes/resultados_carpeta_{i:02}.csv", index=False)
        print(f"Guardado lote {i:02} con {len(df)} archivos.")
        errores = [r for r in resultados if r is not None and 'error' in r]
        if errores:
            with open(f"resultados_lotes/errores_carpeta_{i:02}.txt", 'w') as f:
                for err in errores:
                    f.write(f"{err['archivo']}: {err['error']}\n")
    else:
        print(f"Todos los archivos del lote {i:02} ya estaban procesados.")
