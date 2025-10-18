import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from lib import predict as model_predict

app = FastAPI()


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        print(f'Cargando modelo {model_name} con alias {alias} desde mlflow')
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(
            model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except Exception as error:
        print('Algo salio mal cargando el modelo local', error)
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        print(f'Cargando modelo la metadata desde S3')
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(
            data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(
            data_dictionary["standard_scaler_std"])
    except Exception as error:
        print('Algo salio mal cargando la metadata local', error)
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    print(f'Version del modelo cargada: {version_model_ml}')

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "stroke_prediction_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has
        # changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except BaseException:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the stroke prediction model.
    """
    gender: Literal["Male", "Female", "Other"] = Field(
        description="Género del paciente."
    )
    age: float = Field(
        ge=0,
        le=99,
        description="Edad del paciente en años."
    )
    hypertension: int = Field(
        ge=0,
        le=1,
        description="Indicador de hipertensión (0 = No, 1 = Sí)."
    )
    heart_disease: int = Field(
        ge=0,
        le=1,
        description="Indicador de enfermedad cardíaca (0 = No, 1 = Sí)."
    )
    ever_married: Literal["Yes", "No"] = Field(
        description="Estado civil del paciente."
    )
    work_type: Literal["Private", "Self-employed", "Govt_job", "children",
                       "Never_worked"] = Field(description="Tipo de empleo del paciente.")
    Residence_type: Literal["Urban", "Rural"] = Field(
        description="Tipo de zona residencial (Urbana o Rural)."
    )
    avg_glucose_level: float = Field(
        ge=50,
        le=300,
        description="Nivel promedio de glucosa en sangre."
    )
    bmi: float = Field(
        ge=10,
        le=100,
        description="Índice de Masa Corporal (IMC)."
    )
    smoking_status: Literal["formerly smoked", "never smoked", "smokes"] = Field(
        description="Estado de tabaquismo del paciente.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "age": 67.0,
                    "hypertension": 0,
                    "heart_disease": 1,
                    "ever_married": "Yes",
                    "work_type": "Private",
                    "Residence_type": "Urban",
                    "avg_glucose_level": 228.69,
                    "bmi": 36.6,
                    "smoking_status": "formerly smoked",
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the prediction model.
    """
    prediction: int = Field(description="Prediction of the model")
    description: str = Field(description="Description of the prediction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 1,
                    "description": "The patient is likely to have a stroke.",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model(
    "stroke_prediction_model_prod", "champion")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """
    Root endpoint of the Stroke Prediction API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder(
        {"message": "Welcome to the Stroke Prediction API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting stroke.

    This endpoint receives features related to a patient's health and predicts whether the patient has stroke
    or not using a trained model. It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and
    # dictionary
    features_dict = features.dict()
    prediction = model_predict.run(features_dict, data_dict, model)

    # Convert prediction result into string format
    str_pred = "The patient is healthy."
    if prediction[0] > 0:
        str_pred = "The patient is likely to have a stroke."

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(prediction=prediction[0].item(), description=str_pred)
