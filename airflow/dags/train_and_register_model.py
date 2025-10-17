import datetime

from airflow.decorators import dag, task

markdown_text = """
### Train Process for Stroke Prediction

This DAG trains the model ...
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
    """
    Main DAG function for training and registering the stroke prediction model.
    
    This function orchestrates the model training process using PyCaret,
    evaluates the model, and registers it in the MLflow model registry with
    appropriate versioning and aliasing.
    """


    @task.virtualenv(
        task_id="train_and_register_model_task",
        requirements=[
            "awswrangler==3.6.0",
            "mlflow==2.10.2",
            "scikit-learn>=1.4,<1.7",
            "pycaret @ git+https://github.com/pycaret/pycaret.git@master"
        ],
        system_site_packages=True
    )
    def train_and_register_model_task():
        """
        Train and register the stroke prediction model using PyCaret.
        
        Loads the preprocessed data, trains a Naive Bayes model using PyCaret,
        evaluates it, and registers it in MLflow with champion/challenger logic.
        """


        def _train_model(log):
            """
            Train the stroke prediction model using PyCaret.
            
            Loads training and test data from S3, sets up PyCaret environment,
            trains a Naive Bayes classifier, and returns the model with metrics.
            
            Args:
                log: Logger object for logging training progress.
                
            Returns:
                tuple: (model, accuracy, f1_score) - trained model and its metrics.
            """

            path_X_train = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TRAIN}/X_{consts.TRAIN}.csv"
            path_y_train = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TRAIN}/y_{consts.TRAIN}.csv"
            path_X_test = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TEST}/X_{consts.TEST}.csv"
            path_y_test = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TEST}/y_{consts.TEST}.csv"

            X_train = wr.s3.read_csv(path_X_train)
            y_train = wr.s3.read_csv(path_y_train)
            X_test  = wr.s3.read_csv(path_X_test)
            y_test  = wr.s3.read_csv(path_y_test)

            df_train = X_train.reset_index(drop=True).assign(stroke=y_train.squeeze().values)
            df_test  = X_test.reset_index(drop=True).assign(stroke=y_test.squeeze().values)

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
            model = create_model('nb', cross_validation=False)
            final_model = finalize_model(model)

            log.info("_train_model: model trained")

            metrics = pull()

            log.info(f"_train_model: model metrics:\n{metrics}")

            accuracy = float(metrics.get('Accuracy', [None])[0])
            f1 = float(metrics.get('F1', [None])[0])

            return final_model, accuracy, f1



        def _register_model(log, model, acc, f1):
            """
            Register the trained model in MLflow model registry.
            
            Creates or updates the experiment, logs the model with metrics,
            registers it as a new version, and manages champion/challenger aliases
            based on performance comparison.
            
            Args:
                log: Logger object for logging registration progress.
                model: The trained model object to register.
                acc (float): Accuracy metric of the model.
                f1 (float): F1 score metric of the model.
            """
            log.info("_register_model: started")
            
            def _set_experiment(client, name):
                """
                Create or restore an MLflow experiment.
                
                Args:
                    client: MLflow client instance.
                    name (str): Name of the experiment.
                """
                # 1) Si el experiment ya existe, terminar
                for e in client.search_experiments(view_type=ViewType.ACTIVE_ONLY):
                    if e.name == name:
                        return

                # 2) Si el experiment existe pero está "deleted", restaurarlo y terminar
                deleted = [e for e in client.search_experiments(view_type=ViewType.DELETED_ONLY) if e.name == name]
                if deleted:
                    client.restore_experiment(deleted[0].experiment_id)
                    return

                # 3) Si el experiment no existe, crearlo
                mlflow.set_experiment(name)

            def _safe_float(x, default=0.0):
                """
                Safely convert a value to float with a default fallback.
                
                Args:
                    x: Value to convert to float.
                    default (float): Default value if conversion fails.
                    
                Returns:
                    float: Converted value or default.
                """
                try:
                    return float(x)
                except Exception:
                    return default
                
            def _champion_superado(f1, champion_f1):
                """
                Determine if the challenger surpasses the champion.
                
                Note: Currently simulates the comparison randomly for demonstration purposes.
                In production, this should return (f1 > champion_f1).
                
                Args:
                    f1 (float): F1 score of the challenger model.
                    champion_f1 (float): F1 score of the champion model.
                    
                Returns:
                    bool: True if challenger wins, False otherwise.
                """
                # El champion es superado si f1 > champion_f1. En lugar de retornar esa evaluacion, simulamos de forma aleatoria.
                import random
                return random.choice([True, False])


            EXPERIMENT_NAME = "stroke_training"
            RM_NAME = "stroke_prediction_model_prod"

            mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)
            client = MlflowClient(mlflow.get_tracking_uri())

            _set_experiment(client, EXPERIMENT_NAME)

            # Logueo en MLflow (run + params + métricas + artefacto modelo) y registro como Model Version
            with mlflow.start_run(run_name="nb-stroke-challenger") as run:
                run_id = run.info.run_id

                # Params del estimador (si existen)
                try:
                    params = model.get_params()
                except Exception:
                    params = {}
                for k, v in params.items():
                    mlflow.log_param(str(k), str(v))

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

            # Asegurar Registered Model y ubicar la versión registrada de este run
            client = MlflowClient()
            try:
                client.create_registered_model(name=RM_NAME, description="Stroke classifier (PyCaret/Naive Bayes baseline)")
            except RestException as e:
                if getattr(e, "error_code", "") != "RESOURCE_ALREADY_EXISTS":
                    raise

            # Buscar la Model Version creada por este run
            mv_list = client.search_model_versions(f"name = '{RM_NAME}' and run_id = '{run_id}'")
            if not mv_list:
                # fallback: tomar la última versión registrada
                mv_list = client.get_latest_versions(RM_NAME)
            mv = sorted(mv_list, key=lambda m: int(m.version))[-1]
            challenger_version = mv.version

            # Etiquetas útiles en la versión
            client.set_model_version_tag(RM_NAME, challenger_version, "stage", "challenger")
            client.set_model_version_tag(RM_NAME, challenger_version, "f1", str(f1))
            client.set_model_version_tag(RM_NAME, challenger_version, "accuracy", str(acc))
            client.set_model_version_tag(RM_NAME, challenger_version, "model_class", type(model).__name__)
            for k, v in (params or {}).items():
                client.set_model_version_tag(RM_NAME, challenger_version, str(k), str(v))

            # Reasignar alias 'challenger' a esta versión
            client.set_registered_model_alias(RM_NAME, "challenger", challenger_version)

            # Comparar contra 'champion' (si existe) y promover si mejora
            try:
                champion_mv = client.get_model_version_by_alias(RM_NAME, "champion")
                champion_f1 = _safe_float(champion_mv.tags.get("f1", None), default=0.0)
                log.info(f"_register_model: champion v{champion_mv.version} f1={champion_f1} | challenger v{challenger_version} f1={f1}")

                if _champion_superado(f1, champion_f1):
                    client.set_registered_model_alias(RM_NAME, "champion", challenger_version)
                    log.info(f"_register_model: PROMOTED challenger v{challenger_version} -> CHAMPION")
                else:
                    log.info(f"_register_model: kept CHAMPION v{champion_mv.version}")

            except RestException:
                # No había CHAMPION aún → este challenger pasa a ser el primero
                client.set_registered_model_alias(RM_NAME, "champion", challenger_version)
                log.info(f"_register_model: no champion found — set v{challenger_version} as CHAMPION")

            log.info(f"_register_model: completed (run_id={run_id}, uri={model_uri}, challenger=v{challenger_version})")



        import sys
        sys.path.append("/opt/airflow/dags")
        import logging
        log = logging.getLogger("airflow.task")
        import utils.constants as consts
        import awswrangler as wr
        from pycaret.classification import (
            setup, create_model, finalize_model, pull
        )
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.entities import ViewType
        from mlflow.exceptions import RestException

        log.info("train_and_register_model_task: started")

        model, acc, f1 = _train_model(log)

        log.info("train_and_register_model_task: _train_model executed")

        _register_model(log, model, acc, f1)
        
        log.info("train_and_register_model_task: completed")



    train_and_register_model_task()



dag = train_and_register_model()

