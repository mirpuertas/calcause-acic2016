# Tuning para RF y GP

- Hiperparámetros y metodología utilizada con los 10 datasets provistos por ACIC2016 para el tuning del modelo
- Una vez realizado el tuning se procede a congelar el modelo y se procede a la fase de testing en los datasets black-box

## Tuning para Random Forest

rf_param_grid = {
        'n_estimators':      [50, 100],
        'max_depth':         [4, 6, 8],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

rf_cv = GridSearchCV(
    rf,
    rf_param_grid,
    cv=5,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_jobs=-1
).fit(Xc, yc)


## Tuning para Gaussian-Process

sc = StandardScaler().fit(Xc)
Xc_std = sc.transform(Xc)
Xt_std = sc.transform(Xt)

- RBF con varianza de salida (constante) × RBF
gp_param_grid = {
    'kernel': [
        CK(1.0, (0.1, 10.0)) * RBF(l, (0.1, 10.0))
        for l in [0.5, 1.0, 2.0]               # 3 valores de length-scale
    ],
    'alpha': [1e-3, 3e-3, 1e-2]                           # ruido observado
}

-sub‑muestra controles a 2 000 para la búsqueda
m = min(len(Xc_std), 2000)
idx = np.random.choice(len(Xc_std), m, replace=False)
Xc_sub, yc_sub = Xc_std[idx], yc[idx]

gp = GaussianProcessRegressor(
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=random_state
)

gp_cv = GridSearchCV(
        gp, gp_param_grid,
        cv=3,                                      
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=-1
).fit(Xc_sub, yc_sub)