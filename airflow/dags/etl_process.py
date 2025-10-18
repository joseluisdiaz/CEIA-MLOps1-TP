import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Stroke Prediction Dataset

This DAG extracts information from the Stroke Prediction dataset from Kaggle.
It preprocesses the data by creating dummy variables for categorical columns and scaling numerical features.
After preprocessing, the data is saved back into an S3 bucket (MinIO) as separate CSV files for training and testing.
"""


default_args = {
    'owner': "Group MLOps",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(dag_id="process_etl_stroke_data",
     description="ETL process for stroke data, separating the dataset into training and testing sets.",
     doc_md=markdown_text,
     tags=["ETL",
            "Stroke"],
     default_args=default_args,
     catchup=False,
     )
def process_etl_stroke_data():

    @task.virtualenv(
        task_id="upload_original_csv_if_needed",
        requirements=["awswrangler==3.6.0", "pandas"],
        system_site_packages=True
    )
    def upload_original_csv_if_needed():
        import sys
        sys.path.append("/opt/airflow/dags")
        import awswrangler as wr
        import utils.constants as consts
        import pandas as pd

        local_path = f"{consts.OPT_DATASET}{consts.ORIG_DATA_NAME}"
        s3_path = f"{consts.S3}{consts.BUCKET_RAW}{consts.ORIG_DATA_NAME}"

        try:
            wr.s3.head_object(s3_path)
            print(f"El archivo ya existe en S3: {s3_path}")
        except Exception:
            df = pd.read_csv(local_path)
            # Save data
            wr.s3.to_csv(df=df, path=s3_path, index=False)

            print(f"Archivo subido a S3: {s3_path}")

    @task.virtualenv(
        task_id="null_imputation",
        requirements=["awswrangler==3.6.0", "mlflow==2.10.2"],
        system_site_packages=True
    )
    def null_imputation():
        """
        Impute missing values in the dataset.
        """
        import sys
        sys.path.append("/opt/airflow/dags")
        import pandas as pd
        import awswrangler as wr
        import utils.constants as consts

        data_original_path = f"{consts.S3}{consts.BUCKET_RAW}{consts.ORIG_DATA_NAME}"
        data_end_path = f"{consts.S3}{consts.BUCKET_RAW}{consts.END_DATA_NAME}"
        dataset = wr.s3.read_csv(data_original_path)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        # Imputación de nulos
        col_to_impute = 'bmi'
        col_to_stratify = 'age_group'
        bins = [0, 25, 40, 60, dataset["age"].max() + 1]
        labels = ["0–25", "25–40", "40–60", "60+"]

        dataset["age_group"] = pd.cut(
            dataset["age"], bins=bins, labels=labels, right=False)
        dataset[col_to_impute] = dataset[col_to_impute].fillna(
            dataset.groupby(col_to_stratify)[col_to_impute].transform('median'))
        dataset.drop(columns="age_group", inplace=True)

        wr.s3.to_csv(df=dataset, path=data_end_path, index=False)

    @task.virtualenv(
        task_id="create_mlflow_experiment",
        requirements=["mlflow==2.10.2"],
        system_site_packages=True
    )
    def create_mlflow_experiment():
        import sys
        sys.path.append("/opt/airflow/dags")
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.entities import ViewType
        import utils.constants as consts

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)
        client = MlflowClient(mlflow.get_tracking_uri())
        name = "Stroke"

        # 1) Si el experiment ya existe, terminar
        for e in client.search_experiments(view_type=ViewType.ACTIVE_ONLY):
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

    @task.virtualenv(
        task_id="make_dummies_variables",
        requirements=["awswrangler==3.6.0", "mlflow==2.10.2"],
        system_site_packages=True
    )
    def make_dummies_variables():
        """
        Convert categorical variables into one-hot encoding.
        """
        import sys
        sys.path.append("/opt/airflow/dags")

        import datetime
        import boto3
        import mlflow
        import pandas as pd
        import numpy as np
        from airflow.models import Variable
        import awswrangler as wr

        import utils.etl_utils as etl
        import utils.constants as consts

        data_original_path = f"{consts.S3}{consts.BUCKET_RAW}{consts.ORIG_DATA_NAME}"
        data_end_path = f"{consts.S3}{consts.BUCKET_RAW}{consts.END_DATA_NAME}"

        dataset = wr.s3.read_csv(data_original_path)

        # Generating dummies variables
        with open(f"{consts.OPT_DAGS}{consts.CATEGORICAL_FEATURES_FILE}", "r") as f:
            categorical_features = [line.strip() for line in f if line.strip()]
        dataset[categorical_features] = dataset[categorical_features].astype(
            str)
        dataset_with_dummies = pd.get_dummies(data=dataset,
                                              columns=categorical_features,
                                              drop_first=True)

        wr.s3.to_csv(df=dataset_with_dummies, path=data_end_path, index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = etl.get_metadata_info(client)

        target_col = Variable.get(
            consts.TARGET_COLUMN,
            default_var=consts.TARGET_COLUMN_DEFAULT)
        dataset_log = dataset.drop(columns=target_col)
        dataset_with_dummies_log = dataset_with_dummies.drop(
            columns=target_col)

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_dummy'] = dataset_with_dummies_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = categorical_features
        data_dict['columns_dtypes'] = {
            k: str(v) for k,
            v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_dummy'] = {
            k: str(v) for k, v in dataset_with_dummies_log.dtypes.to_dict().items()}

        category_dummies_dict = {}
        for category in categorical_features:
            category_dummies_dict[category] = np.sort(
                dataset_log[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime(
            '%Y/%m/%d-%H:%M:%S"')

        etl.save_metadata_info(client, data_dict)

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment("Stroke")

        mlflow.start_run(
            run_name='ETL_run_' +
            datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
            experiment_id=experiment.experiment_id,
            tags={
                "experiment": "etl",
                "dataset": "Stroke"},
            log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(
            dataset,
            source=consts.KAGGLE_DATASET_URL,
            targets=target_col,
            name="stroke_data_complete")
        mlflow_dataset_dummies = mlflow.data.from_pandas(
            dataset_with_dummies,
            source=consts.KAGGLE_DATASET_URL,
            targets=target_col,
            name="stroke_data_complete_with_dummies")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import sys
        sys.path.append("/opt/airflow/dags")
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable
        import utils.constants as consts

        data_original_path = f"{consts.S3}{consts.BUCKET_RAW}{consts.END_DATA_NAME}"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = float(Variable.get("test_size_stroke", default_var=0.2))
        target_col = Variable.get("target_col_stroke", default_var="stroke")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y)

        data_final_path = f"{consts.S3}{consts.BUCKET_FINAL}"

        wr.s3.to_csv(
            df=X_train,
            path=f"{data_final_path}{consts.TRAIN}/X_{consts.TRAIN}.csv",
            index=False)
        wr.s3.to_csv(
            df=X_test,
            path=f"{data_final_path}{consts.TEST}/X_{consts.TEST}.csv",
            index=False)
        wr.s3.to_csv(
            df=y_train,
            path=f"{data_final_path}{consts.TRAIN}/y_{consts.TRAIN}.csv",
            index=False)
        wr.s3.to_csv(
            df=y_test,
            path=f"{data_final_path}{consts.TEST}/y_{consts.TEST}.csv",
            index=False)

    @task.virtualenv(
        task_id="normalize_numerical_features",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def normalize_data():
        """
        Standardization of numerical columns
        """
        import sys
        sys.path.append("/opt/airflow/dags")
        import mlflow
        import boto3
        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        import utils.etl_utils as etl
        import utils.constants as consts

        path_train = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TRAIN}/X_{consts.TRAIN}.csv"
        path_test = f"{consts.S3}{consts.BUCKET_FINAL}{consts.TEST}/X_{consts.TEST}.csv"
        X_train = wr.s3.read_csv(path_train)
        X_test = wr.s3.read_csv(path_test)

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        wr.s3.to_csv(df=X_train, path=path_train, index=False)
        wr.s3.to_csv(df=X_test, path=path_test, index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = etl.get_metadata_info(client)

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()

        etl.save_metadata_info(client, data_dict)

        mlflow.set_tracking_uri(consts.MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment("Stroke")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs(
            [experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param(
                "Standard Scaler feature names",
                sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)

    upload_original_csv_if_needed() >> null_imputation() >> create_mlflow_experiment(
    ) >> make_dummies_variables() >> split_dataset() >> normalize_data()


dag = process_etl_stroke_data()
