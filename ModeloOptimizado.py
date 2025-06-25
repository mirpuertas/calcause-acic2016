from sklearn.ensemble  import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np, pandas as pd


def calcause(data: pd.DataFrame,
             covariates: list,
             n_bootstrap: int = 0,
             random_state: int = 0):
    """
    CalCause estilo ACIC‑2016 **sin nueva búsqueda de hiper‑parámetros**:
        • Random Forest     – fijos (n=100, depth=8, min_split=5)
        • Gaussian Process  – kernel =  CK×RBF(ℓ=1), α = 0.01
        • 3‑fold CV en el grupo control para decidir RF vs GP
        • Se re‑entrena el ganador con todo el grupo control
        • Devuelve SATT, IC bootstrap y métricas básicas
    """
    rng = np.random.default_rng(random_state)

    tr   = data[data.z == 1]
    ctl  = data[data.z == 0]

    Xc, yc = ctl[covariates].values,  ctl.y.values
    Xt, yt = tr [covariates].values,  tr .y.values

    # Paso 1: Planteo de los modelos con los hiperparámetros fijos (Fijados por tuning previo)
    rf_params = dict(n_estimators=100, max_depth=8, min_samples_split=5,
                     random_state=random_state, n_jobs=-1)
    rf = RandomForestRegressor(**rf_params)


    kernel = CK(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 10.0)) # Kernel con HPs iniciales
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=1e-2,         
                                  normalize_y=True,
                                  n_restarts_optimizer=0, # IMPORTANTE: Usa HPs iniciales del kernel, no optimiza.
                                  random_state=random_state)

    # Paso 2: Evaluación de los modelos con CV en el grupo control (3 folds)
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)


    rf_mse = -cross_val_score(rf, Xc, yc,
                              cv=cv,
                              scoring='neg_mean_squared_error',
                              n_jobs=-1).mean()

    
    sc     = StandardScaler().fit(Xc) # IMPORTANTE: GP necesita estandarizar X
    Xc_std = sc.transform(Xc)
    Xt_std = sc.transform(Xt)

    gp_mse = -cross_val_score(gp, Xc_std, yc,
                              cv=cv,
                              scoring='neg_mean_squared_error',
                              n_jobs=-1).mean()

    # Paso 3: Selección del modelo y ajuste final sobre todo el grupo control
    if gp_mse < rf_mse:                           
        modelo = "GaussianProcess"
        gp.fit(Xc_std, yc)
        y0_hat = gp.predict(Xt_std)
    else:                                     
        modelo = "RandomForest"
        rf.fit(Xc, yc)
        y0_hat = rf.predict(Xt)

    # Paso 4: Cálculo del SATT y bootstrap
    ite  = yt - y0_hat
    satt = float(np.mean(ite))

    ci_95 = ci_ancho = None
    if n_bootstrap:
        n = len(ite)
        idx = rng.integers(0, n, (n_bootstrap, n))
        satt_boot = ite[idx].mean(axis=1)
        ci_95 = np.percentile(satt_boot, [2.5, 97.5])
        ci_ancho = ci_95[1] - ci_95[0]

    return dict(
        satt           = satt,
        ite_hat       = ite,
        ci_95          = ci_95,
        ci_ancho       = ci_ancho,
        modelo_elegido = modelo,
        mse_rf         = rf_mse,
        mse_gp         = gp_mse,
        params_rf      = rf_params,
        params_gp      = {"kernel": str(kernel), "alpha": 1e-2}
    )
