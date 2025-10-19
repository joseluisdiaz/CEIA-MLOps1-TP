import datetime

from airflow.decorators import dag, task

markdown_text = """
### Train Process for Stroke Prediction

This DAG trains multiple models for stroke prediction with automated hyperparameter optimization:

#### Models:
- **Naive Bayes**: Optimized using PyCaret's tune_model with Optuna backend
- **XGBoost**: Optimized using custom Optuna implementation

#### Configuration Options:
- **N_TRIALS_XGBOOST**: Number of Optuna trials for XGBoost (default: 10)
- **N_TRIALS_NB**: Number of tuning iterations for Naive Bayes (default: 10)

#### Features:
- Automated hyperparameter optimization for BOTH models
- Naive Bayes: PyCaret tune_model with TPE algorithm
- XGBoost: Custom Optuna optimization with MedianPruner
- Model versioning and registration in MLflow
- Champion/Challenger model comparison
- Comprehensive metrics logging (F1, Accuracy)
- All hyperparameters logged and tracked
"""

default_args = {
    'owner': "Group MLOps",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="train_and_register_model",
    description="Train process for Stricke Prediction.",
    doc_md=markdown_text,
    tags=["Train", "Stroke", "Register"],
    default_args=default_args,
    catchup=False,
)
def train_and_register_model():

    @task.virtualenv(
        task_id="train_and_register_model_task",
        requirements=[
            "awswrangler==3.6.0",
            "mlflow==2.10.2",
            "scikit-learn>=1.4,<1.7",
            "xgboost==2.0.3",
            "optuna==3.6.1",
            "pycaret @ git+https://github.com/pycaret/pycaret.git@master"
        ],
        system_site_packages=True
    )
    def train_and_register_model_task():

        def _optimize_xgboost_hyperparams(X, y, n_trials=10):
            """
            Optimize XGBoost hyperparameters using Optuna
            """
            import optuna
            from optuna.pruners import MedianPruner
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 3),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
                }
                clf = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
                score = cross_val_score(
                    clf, X, y, cv=5, scoring='f1', n_jobs=-1).mean()

                # Reportamos para el pruning
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize",
                pruner=MedianPruner(n_warmup_steps=5)
            )
            study.optimize(objective, n_trials=n_trials)

            return study.best_params

        def _train_model(log, n_trials_xgb=10, n_trials_nb=10):

            path_X_train = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TRAIN}/X_{consts.TRAIN}.csv"
            path_y_train = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TRAIN}/y_{consts.TRAIN}.csv"
            path_X_test = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TEST}/X_{consts.TEST}.csv"
            path_y_test = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TEST}/y_{consts.TEST}.csv"

            X_train = wr.s3.read_csv(path_X_train)
            y_train = wr.s3.read_csv(path_y_train)
            X_test = wr.s3.read_csv(path_X_test)
            y_test = wr.s3.read_csv(path_y_test)

            df_train = X_train.reset_index(
                drop=True).assign(
                stroke=y_train.squeeze().values)
            df_test = X_test.reset_index(
                drop=True).assign(
                stroke=y_test.squeeze().values)

            log.info("_train_model: datasets loaded")

            setup(
                data=df_train,
                test_data=df_test,
                target='stroke',
                session_id=42,
                preprocess=True,
                normalize=False,
                verbose=True,
                index=False
            )

            # Train and tune Naive Bayes model
            log.info(
                f"_train_model: Starting Naive Bayes hyperparameter tuning with {n_trials_nb} iterations")
            nb_model = create_model('nb')
            tuned_nb_model = tune_model(
                nb_model,
                n_iter=n_trials_nb,
                optimize='F1',
                search_library='optuna',
                search_algorithm='tpe'
            )
            final_nb_model = finalize_model(tuned_nb_model)
            nb_metrics = pull()
            nb_accuracy = float(nb_metrics.get('Accuracy', [None])[0])
            nb_f1 = float(nb_metrics.get('F1', [None])[0])

            # Obtener los mejores hiperparámetros de Naive Bayes
            try:
                nb_params = tuned_nb_model.get_params()
            except Exception:
                nb_params = {}

            log.info(
                f"_train_model: Naive Bayes tuned - Accuracy: {nb_accuracy}, F1: {nb_f1}")
            log.info(f"_train_model: Best Naive Bayes params: {nb_params}")

            # Train XGBoost model with Optuna optimization
            log.info(
                f"_train_model: Starting XGBoost Optuna optimization with {n_trials_xgb} trials")
            xgb_params = _optimize_xgboost_hyperparams(
                X_train, y_train.squeeze(), n_trials=n_trials_xgb)
            log.info(
                f"_train_model: Best XGBoost hyperparameters found: {xgb_params}")

            xgb_model = create_model('xgboost', **xgb_params)
            final_xgb_model = finalize_model(xgb_model)
            xgb_metrics = pull()
            xgb_accuracy = float(xgb_metrics.get('Accuracy', [None])[0])
            xgb_f1 = float(xgb_metrics.get('F1', [None])[0])

            log.info(
                f"_train_model: XGBoost trained - Accuracy: {xgb_accuracy}, F1: {xgb_f1}")

            return {
                'nb': (final_nb_model, nb_accuracy, nb_f1, nb_params),
                'xgb': (final_xgb_model, xgb_accuracy, xgb_f1, xgb_params)
            }

        def _register_model(
                log,
                model,
                acc,
                f1,
                model_type='nb',
                hyperparams=None):
            log.info(f"_register_model: started for {model_type}")

            def _set_experiment(client, name):
                # 1) Si el experiment ya existe, terminar
                for e in client.search_experiments(
                        view_type=ViewType.ACTIVE_ONLY):
                    if e.name == name:
                        return

                # 2) Si el experiment existe pero está "deleted", restaurarlo y
                # terminar
                deleted = [e for e in client.search_experiments(
                    view_type=ViewType.DELETED_ONLY) if e.name == name]
                if deleted:
                    client.restore_experiment(deleted[0].experiment_id)
                    return

                # 3) Si el experiment no existe, crearlo
                mlflow.set_experiment(name)

            def _safe_float(x, default=0.0):
                try:
                    return float(x)
                except Exception:
                    return default

            def _champion_superado(f1, champion_f1):
                # El champion es superado si f1 > champion_f1. En lugar de
                # retornar esa evaluacion, simulamos de forma aleatoria.
                import random
                return random.choice([True, False])

            EXPERIMENT_NAME = "stroke_training_5"
            RM_NAME = "stroke_prediction_model_prod"

            mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)
            client = MlflowClient(mlflow.get_tracking_uri())

            _set_experiment(client, EXPERIMENT_NAME)

            # Logueo en MLflow (run + params + métricas + artefacto modelo) y
            # registro como Model Version
            run_name = f"{model_type}-stroke-challenger"
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Params del estimador (si existen)
                try:
                    params = model.get_params()
                except Exception:
                    params = {}
                for k, v in params.items():
                    mlflow.log_param(str(k), str(v))

                # Log hyperparameters adicionales si existen
                if hyperparams:
                    for k, v in hyperparams.items():
                        mlflow.log_param(f"optuna_{k}", str(v))

                # Métricas
                mlflow.log_metric("accuracy", float(acc))
                mlflow.log_metric("f1", float(f1))

                # Log del modelo (scikit-learn) y registro en RM_NAME
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=RM_NAME
                )
                model_uri = f"runs:/{run_id}/model"

            # Asegurar Registered Model y ubicar la versión registrada de este
            # run
            client = MlflowClient()
            try:
                client.create_registered_model(
                    name=RM_NAME, description="Stroke classifier (PyCaret/Naive Bayes baseline)")
            except RestException as e:
                if getattr(e, "error_code", "") != "RESOURCE_ALREADY_EXISTS":
                    raise

            # Buscar la Model Version creada por este run
            mv_list = client.search_model_versions(
                f"name = '{RM_NAME}' and run_id = '{run_id}'")
            if not mv_list:
                # fallback: tomar la última versión registrada
                mv_list = client.get_latest_versions(RM_NAME)
            mv = sorted(mv_list, key=lambda m: int(m.version))[-1]
            challenger_version = mv.version

            # Etiquetas útiles en la versión
            client.set_model_version_tag(
                RM_NAME, challenger_version, "stage", "challenger")
            client.set_model_version_tag(
                RM_NAME, challenger_version, "f1", str(f1))
            client.set_model_version_tag(
                RM_NAME, challenger_version, "accuracy", str(acc))
            client.set_model_version_tag(
                RM_NAME,
                challenger_version,
                "model_class",
                type(model).__name__)
            for k, v in (params or {}).items():
                client.set_model_version_tag(
                    RM_NAME, challenger_version, str(k), str(v))

            # Agregar hyperparameters como tags si existen
            if hyperparams:
                for k, v in hyperparams.items():
                    client.set_model_version_tag(
                        RM_NAME, challenger_version, f"optuna_{k}", str(v))

            # Reasignar alias 'challenger' a esta versión
            client.set_registered_model_alias(
                RM_NAME, "challenger", challenger_version)

            # Comparar contra 'champion' (si existe) y promover si mejora
            try:
                champion_mv = client.get_model_version_by_alias(
                    RM_NAME, "champion")
                champion_f1 = _safe_float(
                    champion_mv.tags.get(
                        "f1", None), default=0.0)
                log.info(
                    f"_register_model: champion v{champion_mv.version} f1={champion_f1} | challenger v{challenger_version} f1={f1}")

                if _champion_superado(f1, champion_f1):
                    client.set_registered_model_alias(
                        RM_NAME, "champion", challenger_version)
                    log.info(
                        f"_register_model: PROMOTED challenger v{challenger_version} -> CHAMPION")
                else:
                    log.info(
                        f"_register_model: kept CHAMPION v{champion_mv.version}")

            except RestException:
                # No había CHAMPION aún → este challenger pasa a ser el primero
                client.set_registered_model_alias(
                    RM_NAME, "champion", challenger_version)
                log.info(
                    f"_register_model: no champion found — set v{challenger_version} as CHAMPION")

            log.info(
                f"_register_model: completed (run_id={run_id}, uri={model_uri}, challenger=v{challenger_version})")

        import sys
        sys.path.append("/opt/airflow/dags")
        import logging
        log = logging.getLogger("airflow.task")
        import utils.constants as consts
        import awswrangler as wr
        from pycaret.classification import (
            setup, create_model, finalize_model, pull, tune_model
        )
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.entities import ViewType
        from mlflow.exceptions import RestException

        log.info("train_and_register_model_task: started")

        # Parámetros de configuración - Número de iteraciones para optimización
        N_TRIALS_XGBOOST = 10  # Número de trials de Optuna para XGBoost
        N_TRIALS_NB = 10       # Número de iteraciones de tuning para Naive Bayes

        models_dict = _train_model(
            log,
            n_trials_xgb=N_TRIALS_XGBOOST,
            n_trials_nb=N_TRIALS_NB)

        log.info("train_and_register_model_task: _train_model executed")

        # Register Naive Bayes model
        nb_model, nb_acc, nb_f1, nb_hyperparams = models_dict['nb']
        _register_model(
            log,
            nb_model,
            nb_acc,
            nb_f1,
            model_type='nb',
            hyperparams=nb_hyperparams)
        log.info("train_and_register_model_task: Naive Bayes registered")

        # Register XGBoost model
        xgb_model, xgb_acc, xgb_f1, xgb_hyperparams = models_dict['xgb']
        _register_model(
            log,
            xgb_model,
            xgb_acc,
            xgb_f1,
            model_type='xgboost',
            hyperparams=xgb_hyperparams)
        log.info("train_and_register_model_task: XGBoost registered")

        log.info("train_and_register_model_task: completed")

    train_and_register_model_task()


dag = train_and_register_model()
