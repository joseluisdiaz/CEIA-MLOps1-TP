# Implementación de un Modelo de Predicción de Accidentes Cerebrovasculares (Stroke)

### MLOPS1 - CEIA - FIUBA

En este proyecto, mostramos una implementación de un modelo productivo para predecir si un paciente tiene riesgo de sufrir un accidente cerebrovascular (ACV), utilizando prácticas de MLOps. Para ello, obtenemos los datos del [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

La implementación incluye:

* En **Apache Airflow**, un DAG (`process_etl_stroke_data`) que obtiene los datos del dataset, realiza limpieza de datos, imputación de valores nulos, feature engineering con variables dummy, y guarda en el bucket `s3://data` los datos separados para entrenamiento y pruebas. **MLflow** hace seguimiento de este procesamiento ETL.

* En **Apache Airflow**, un DAG (`train_and_register_model`) que entrena un modelo de clasificación utilizando **PyCaret** con un algoritmo Naive Bayes. El modelo se evalúa y se registra en el registro de modelos de MLflow con métricas de accuracy y F1-score.
Además, este DAG reentrena el modelo dado un nuevo conjunto de datos. 
Se implementa el paradigma **champion-challenger**: el nuevo modelo se compara con el mejor modelo actual (llamado `champion`), y si obtiene un mejor F1-score, se promociona como el nuevo champion. Todo se registra en MLflow.

* Una **notebook de experimentación** (`experiment_mlflow.ipynb`) para ejecutar localmente con **Optuna**, que realiza una búsqueda de hiperparámetros y encuentra el mejor modelo. Todo el experimento se registra en MLflow, incluyendo gráficos de importancia de características.

* Un **servicio de API** del modelo (FastAPI), que toma el artefacto de MLflow y lo expone para realizar predicciones en tiempo real.

![Diagrama de servicios](example_project.png)

Las flechas verdes y violetas representan las conexiones entre los diferentes servicios del sistema.

---

## Integrantes del Proyecto

- Carla Espínola (carlae.hamm@gmail.com)
- Antonella Gambarte (antonellagambarte@gmail.com)
- Daniela Putrino (dsputrino@gmail.com)
- José Luis Diaz (diazjoseluis@gmail.com)
- Ricardo Silvera (rsilvera@thalu.com.ar)
- José Aviani (jose.aviani@gmail.com)

---

## Testeo de Funcionamiento

El orden para probar el funcionamiento completo es el siguiente:

1. **Levantar el sistema**: 
    a. Ejecuta `docker compose build airflow-worker airflow-scheduler airflow-apiserver airflow-dag-processor` para construir las imágenes Docker para los distintos servicios que forman parte del despliegue de Apache Airflow.
    b. Ejecuta `docker compose --profile all up` para iniciar todos los servicios (Airflow, MLflow, MinIO, PostgreSQL, FastAPI).

2. **Ejecutar el ETL**: Tan pronto como se levante el sistema multi-contenedor, ejecuta en Airflow el DAG llamado `process_etl_stroke_data`. De esta manera se crearán los datos procesados en el bucket `s3://data`.

3. **Entrenar el modelo inicial**: Ejecuta el DAG `train_and_register_model` en Airflow para entrenar el primer modelo y registrarlo como champion en MLflow. Alternativamente, puedes ejecutar la notebook `experiment_mlflow.ipynb` (ubicada en `notebook_example`) para realizar la búsqueda de hiperparámetros con Optuna y entrenar el mejor modelo.

4. **Utilizar el servicio de API**: Una vez entrenado el modelo, utiliza el servicio de API para realizar predicciones.

5. **Reentrenar el modelo (opcional)**: Una vez entrenado el modelo champion, puedes ejecutar el DAG `train_and_register_model` para entrenar un modelo challenger que compita con el campeón. Este DAG automáticamente evaluará ambos modelos y promocionará al challenger si supera al champion en F1-score.

---

## API de Predicción

Podemos realizar predicciones utilizando la API, accediendo a `http://localhost:8800/`.

Para hacer una predicción, debemos enviar una solicitud al endpoint `predict/` con un cuerpo de tipo JSON que contenga un campo de características (`features`) con cada entrada para el modelo.

### Ejemplo de Uso con `curl`:

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "gender_Male": 1,
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married_Yes": 1,
    "work_type_Private": 1,
    "work_type_Self-employed": 0,
    "work_type_children": 0,
    "Residence_type_Urban": 1,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status_formerly smoked": 1,
    "smoking_status_never smoked": 0,
    "smoking_status_smokes": 0
  }
}'
```

### Respuesta del Modelo:

La respuesta del modelo será un valor booleano y un mensaje en forma de cadena de texto que indicará si el paciente tiene riesgo de sufrir un ACV.

```json
{
  "int_output": 1,
  "str_output": "Stroke risk detected"
}
```

Para obtener más detalles sobre la API y probar los endpoints interactivamente, ingresa a `http://localhost:8800/docs`.

> **Nota**: Recuerda que si esto se ejecuta en un servidor diferente a tu computadora, debes reemplazar `localhost` por la IP correspondiente o el dominio DNS, si corresponde.

---

## Características del Dataset

El dataset de Stroke Prediction contiene las siguientes características:

- **gender**: Género del paciente (Male/Female/Other)
- **age**: Edad del paciente
- **hypertension**: Si el paciente tiene hipertensión (0: No, 1: Sí)
- **heart_disease**: Si el paciente tiene enfermedad cardíaca (0: No, 1: Sí)
- **ever_married**: Si el paciente ha estado casado (Yes/No)
- **work_type**: Tipo de trabajo (Private, Self-employed, Govt_job, children, Never_worked)
- **Residence_type**: Tipo de residencia (Urban/Rural)
- **avg_glucose_level**: Nivel promedio de glucosa en sangre
- **bmi**: Índice de masa corporal
- **smoking_status**: Estado de fumador (formerly smoked, never smoked, smokes, Unknown)
- **stroke**: Variable objetivo - Si el paciente sufrió un ACV (0: No, 1: Sí)

---

## Arquitectura del Sistema

El proyecto utiliza los siguientes componentes:

- **Apache Airflow**: Orquestación de pipelines ETL y entrenamiento de modelos
- **MLflow**: Tracking de experimentos, registro de modelos y gestión de versiones
- **MinIO**: Almacenamiento de objetos S3-compatible para datos y artefactos
- **PostgreSQL**: Base de datos para metadatos de Airflow y MLflow
- **FastAPI**: API REST para servir predicciones del modelo
- **PyCaret**: Framework de AutoML para entrenamiento simplificado de modelos
- **Docker Compose**: Orquestación de todos los servicios

---

## Tecnologías Utilizadas

- Python 3.10+
- Apache Airflow 2.8+
- MLflow 2.10+
- FastAPI
- PyCaret
- Scikit-learn
- Pandas
- Docker & Docker Compose
