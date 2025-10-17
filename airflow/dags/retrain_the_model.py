import datetime

from airflow.decorators import dag, task
import utils.constants as consts

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
    """
    Main DAG function for model retraining and evaluation.
    
    Orchestrates the challenger-champion model paradigm:
    - Trains a new challenger model
    - Evaluates both champion and challenger models
    - Promotes the challenger to champion if it performs better
    """

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_the_challenger_model():
        """
        Train a new challenger model based on the current champion model architecture.
        
        Clones the champion model, trains it on the current training data,
        evaluates it on test data, and registers it as a challenger in MLflow.
        """
        import datetime
        import mlflow
        import awswrangler as wr

        from sklearn.base import clone
        from sklearn.metrics import f1_score
        from mlflow.models import infer_signature
        import utils.etl_utils as etl

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)

        def load_the_champion_model():
            """
            Load the current champion model from MLflow model registry.
            
            Returns:
                The champion model object loaded from MLflow.
            """
            model_name = "stroke_detection_model_prod"
            alias = "champion"

            champion_version = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

            return champion_version

        def load_the_train_test_data():
            """
            Load training and test datasets from S3.
            
            Returns:
                tuple: (X_train, y_train, X_test, y_test) dataframes.
            """
            data_final_path = f"{consts.S3}{consts.BUCKET}/{consts.DATA_FINAL_PATH}"
            X_train = etl.load_data(f"{data_final_path}/{consts.TRAIN}/X_train.csv")
            y_train = etl.load_data(f"{data_final_path}/{consts.TRAIN}/y_train.csv")
            X_test = etl.load_data(f"{data_final_path}/{consts.TEST}/X_test.csv")
            y_test = etl.load_data(f"{data_final_path}/{consts.TEST}/y_test.csv")

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            """
            Track the model training experiment in MLflow.
            
            Logs model parameters, creates a new run, and saves the model artifact.
            
            Args:
                model: The trained scikit-learn model.
                X: Training features for signature inference.
                
            Returns:
                str: The MLflow model URI for the logged model.
            """
            # Track the experiment
            experiment = mlflow.set_experiment("Stroke Detection")

            with mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                                    experiment_id=experiment.experiment_id,
                                    tags={"experiment": "challenger models", "dataset": "Stroke Detection"},
                                    log_system_metrics=True
            ) as run:

                params = model.get_params()
                params["model"] = type(model).__name__

                mlflow.log_params(params)

                # Save the artifact of the challenger model
                artifact_path = "model"

                signature = infer_signature(X, model.predict(X))

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    serialization_format='cloudpickle',
                    registered_model_name="stroke_detection_model_dev",
                    metadata={"model_data_version": 1}
                )

                # Obtain the model URI
                model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                return model_uri


        def register_challenger(model, f1_score, model_uri):
            """
            Register the trained model as a challenger in the model registry.
            
            Creates a new model version with tags for model parameters and metrics,
            and assigns the 'challenger' alias to this version.
            
            Args:
                model: The trained model object.
                f1_score: The F1 score of the model on test data.
                model_uri: The MLflow URI of the logged model.
            """
            client = mlflow.MlflowClient()
            name = "stroke_detection_model_prod"

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["f1-score"] = f1_score

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[1],
                tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(name, "challenger", result.version)

        # Load the champion model
        champion_model = load_the_champion_model()

        # Clone the model
        challenger_model = clone(champion_model)

        # Load the dataset
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        # Fit the training model
        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # Obtain the metric of the model
        y_pred = challenger_model.predict(X_test)
        f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)

        # Record the model
        register_challenger(challenger_model, f1_score, artifact_uri)


    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def evaluate_champion_challenge():
        """
        Evaluate and compare the champion and challenger models.
        
        Loads both models, evaluates them on test data, logs metrics to MLflow,
        and promotes the challenger to champion if it performs better based on F1 score.
        """
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import f1_score
        import etl_process as etl

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)

        def load_the_model(alias):
            """
            Load a model from the registry by its alias.
            
            Args:
                alias (str): The model alias ('champion' or 'challenger').
                
            Returns:
                The loaded model object.
            """
            model_name = "stroke_detection_model_prod"

            model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

            return model

        def load_the_test_data():
            """
            Load test dataset from S3.
            
            Returns:
                tuple: (X_test, y_test) dataframes.
            """
            data_final_path = f"{consts.S3}{consts.BUCKET}/{consts.DATA_FINAL_PATH}"
            X_test = etl.load_data(f"{data_final_path}/{consts.TEST}/X_{consts.TEST}.csv")
            y_test = etl.load_data(f"{data_final_path}/{consts.TEST}/y_{consts.TEST}.csv")

            return X_test, y_test

        def promote_challenger(name):
            """
            Promote the challenger model to champion status.
            
            Removes the 'champion' alias from the current champion, removes the
            'challenger' alias from the challenger, and assigns 'champion' to
            the former challenger.
            
            Args:
                name (str): The registered model name.
            """
            client = mlflow.MlflowClient()

            # Demote the champion
            client.delete_registered_model_alias(name, "champion")

            # Load the challenger from registry
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

            # Transform in champion
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):
            """
            Demote the challenger model by removing its alias.
            
            Removes the 'challenger' alias, effectively discarding the model
            as it did not outperform the current champion.
            
            Args:
                name (str): The registered model name.
            """
            client = mlflow.MlflowClient()

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

        # Load the champion model
        champion_model = load_the_model("champion")

        # Load the challenger model
        challenger_model = load_the_model("challenger")

        # Load the dataset
        X_test, y_test = load_the_test_data()

        # Obtain the metric of the models
        y_pred_champion = champion_model.predict(X_test)
        f1_score_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion)

        y_pred_challenger = challenger_model.predict(X_test)
        f1_score_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger)

        experiment = mlflow.set_experiment("Stroke Detection")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_f1_challenger", f1_score_challenger)
            mlflow.log_metric("test_f1_champion", f1_score_champion)

            if f1_score_challenger > f1_score_champion:
                mlflow.log_param("Winner", 'Challenger')
            else:
                mlflow.log_param("Winner", 'Champion')

        name = "stroke_detection_model_prod"
        if f1_score_challenger > f1_score_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
