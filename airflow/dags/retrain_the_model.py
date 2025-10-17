import datetime

from airflow.decorators import dag, task

markdown_text = """
### Re-Train the Model for Stroke Prediction Data

This DAG re-trains the model based on new data, tests the previous model, and put in production the new one 
if it performs  better than the old one. It uses the F1 score to evaluate the model with the test data.

"""

default_args = {
    'owner': "AlumnosOct25",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="retrain_the_model",
    description="Re-train the model based on new data, tests the previous model, and put in production the new one if "
                "it performs better than the old one",
    doc_md=markdown_text,
    tags=["Re-Train", "Stroke"],
    default_args=default_args,
    catchup=False,
)
def processing_dag():
    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=[
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
    def train_the_challenger_model():
        import sys
        sys.path.append("/opt/airflow/dags")
        import datetime as _dt
        import mlflow
        import awswrangler as wr
        import utils.constants as consts
        from sklearn.base import clone
        from sklearn.metrics import f1_score as f1_metric
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)

        def load_the_champion_model():
            model_name = "stroke_prediction_model_prod"
            alias = "champion"
            return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

        def load_the_train_test_data():
            base = f"{consts.S3}{consts.BUCKET_FINAL}"
            X_train = wr.s3.read_csv(f"{base}{consts.TRAIN}/X_{consts.TRAIN}.csv")
            y_train = wr.s3.read_csv(f"{base}{consts.TRAIN}/y_{consts.TRAIN}.csv")
            X_test = wr.s3.read_csv(f"{base}{consts.TEST}/X_{consts.TEST}.csv")
            y_test = wr.s3.read_csv(f"{base}{consts.TEST}/y_{consts.TEST}.csv")
            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            experiment = mlflow.set_experiment("stroke_training")
            with mlflow.start_run(
                run_name='Challenger_run_' + _dt.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": "Stroke Detection"},
                log_system_metrics=True
            ) as run:
                params = {}
                try:
                    params = model.get_params()
                except Exception:
                    pass
                params["model"] = type(model).__name__
                mlflow.log_params({k: str(v) for k, v in params.items()})

                artifact_path = "model"
                try:
                    signature = infer_signature(X, model.predict(X))
                except Exception:
                    signature = None

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    serialization_format='cloudpickle',
                    registered_model_name=None,
                    metadata={"model_data_version": 1}
                )
                return f"runs:/{run.info.run_id}/{artifact_path}"

        def register_challenger(model, f1_value, model_uri):
            client = mlflow.MlflowClient()
            name = "stroke_prediction_model_prod"
            tags = {}
            try:
                tags = model.get_params()
            except Exception:
                pass
            tags["model"] = type(model).__name__
            tags["f1-score"] = float(f1_value)
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[1],
                tags={k: str(v) for k, v in tags.items()}
            )
            client.set_registered_model_alias(name, "challenger", result.version)

        champion_model = load_the_champion_model()
        challenger_model = clone(champion_model)
        X_train, y_train, X_test, y_test = load_the_train_test_data()
        challenger_model.fit(X_train, y_train.to_numpy().ravel())
        y_pred = challenger_model.predict(X_test)
        f1 = f1_metric(y_test.to_numpy().ravel(), y_pred)
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)
        register_challenger(challenger_model, f1, artifact_uri)

    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=[
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
    def evaluate_champion_challenge():
        import sys
        sys.path.append("/opt/airflow/dags")
        import mlflow
        import awswrangler as wr
        import utils.constants as consts
        from sklearn.metrics import f1_score as f1_metric

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)

        def load_the_model(alias):
            model_name = "stroke_prediction_model_prod"
            return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

        def load_the_test_data():
            base = f"{consts.S3}{consts.BUCKET_FINAL}"
            X_test = wr.s3.read_csv(f"{base}{consts.TEST}/X_{consts.TEST}.csv")
            y_test = wr.s3.read_csv(f"{base}{consts.TEST}/y_{consts.TEST}.csv")
            return X_test, y_test

        def promote_challenger(name):
            client = mlflow.MlflowClient()
            # Remove champion alias
            try:
                client.delete_registered_model_alias(name, "champion")
            except Exception:
                pass
            challenger_version = client.get_model_version_by_alias(name, "challenger")
            # Remove challenger alias and promote
            client.delete_registered_model_alias(name, "challenger")
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):
            client = mlflow.MlflowClient()
            try:
                client.delete_registered_model_alias(name, "challenger")
            except Exception:
                pass

        champion_model = load_the_model("champion")
        challenger_model = load_the_model("challenger")
        X_test, y_test = load_the_test_data()
        y_pred_champion = champion_model.predict(X_test)
        f1_score_champion = f1_metric(y_test.to_numpy().ravel(), y_pred_champion)
        y_pred_challenger = challenger_model.predict(X_test)
        f1_score_challenger = f1_metric(y_test.to_numpy().ravel(), y_pred_challenger)

        experiment = mlflow.set_experiment("stroke_training")
        runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
        if runs:
            with mlflow.start_run(run_id=runs[0].info.run_id):
                mlflow.log_metric("test_f1_challenger", float(f1_score_challenger))
                mlflow.log_metric("test_f1_champion", float(f1_score_champion))
                mlflow.log_param("Winner", 'Challenger' if f1_score_challenger > f1_score_champion else 'Champion')

        name = "stroke_prediction_model_prod"
        if f1_score_challenger > f1_score_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
