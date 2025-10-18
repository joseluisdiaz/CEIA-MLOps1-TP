# MLOps - Predictor de Accidente Cerebrovascular

## Trabajo Práctico - CEIA - FIUBA

<br />

**Integrantes:**
- Carla Espínola (carlae.hamm@gmail.com)
- Antonella Gambarte (antonellagambarte@gmail.com)
- Daniela Putrino	(dsputrino@gmail.com)
- José Luis Diaz (diazjoseluis@gmail.com)
- Ricardo Silvera (rsilvera@thalu.com.ar)
- José Aviani (jose.aviani@gmail.com)

<br />

## Inicio rápido

### Opción 1: Script Automático (Recomendado)
```bash
# Ejecutar el script de inicialización
./iniciar_sistema.sh
```

### Opción 2: Docker Compose Manual
```bash
# Iniciar todo el sistema
docker compose --profile all up --build -d

# Ver estado
docker compose ps
```

###  Acceso a Servicios
- **🖥️ Frontend Web**: http://localhost:3000
- ** API FastAPI**: http://localhost:8800
- ** Airflow**: http://localhost:8080 (admin/admin)
- ** MLflow**: http://localhost:5000 (airflow/airflow)
- ** MinIO**: http://localhost:9001 (minio/minio123)

<br />

## Descripción del Proyecto

Este es un sistema **MLOps completo** para la predicción de accidentes cerebrovasculares (ACV). Implementa un pipeline end-to-end que incluye:

- **ETL automatizado** con Apache Airflow
- **Experimentación** y tracking con MLflow
- **API de predicción** con FastAPI
- **Frontend web interactivo** para usuarios finales
- **Almacenamiento distribuido** con MinIO (S3-compatible)
- **Containerización completa** con Docker

### Modelo de Predicción
Utilizamos el [Stroke Prediction Dataset de Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) para entrenar un modelo que predice la probabilidad de ACV basado en:

- Información demográfica (edad, sexo)
- Indicadores médicos (presión arterial, colesterol, glucosa)
- Factores de riesgo (tabaquismo, enfermedades cardíacas)
- Resultados de estudios (ECG, angiografía)

<br />

## Arquitectura del Sistema

<< aquí va el Gráfico qu está haciendo Anto >>

<br />

## Componentes Principales

### 1. **Frontend Web** (`/frontend/`)
- **Tecnología**: HTML5, CSS3, JavaScript vanilla
- **Funcionalidades**:
  - Formulario para ingresar los datos del paciente
  - Validación 
  - Visualización de resultados
  - Datos de ejemplo para testing
  - Diseño responsive

### 2. **API de Predicción** (`/dockerfiles/fastapi/`)
- **Tecnología**: FastAPI, MLflow Client
- **Funcionalidades**:
  - Endpoint REST para predicciones
  - Validación 
  - Carga automática de modelos desde MLflow
  - Sistema Champion/Challenger
  - Documentación automática (Swagger) ??

### 3. **Pipeline ETL** (`/airflow/dags/`)
- **Tecnología**: Apache Airflow
- **Funcionalidades**:
  - Carga de datos desde S3
  - Limpieza e imputación de valores faltantes
  - Creación de variables dummy
  - Normalización de características
  - División train/test estratificada

### 4. **Experimentación** (`/notebook_example/`)
- **Tecnología**: MLflow, Optuna
- **Funcionalidades**:
  - Optimización bayesiana de hiperparámetros
  - Registro automático de experimentos
  - Comparación de múltiples algoritmos
  - Validación cruzada con métricas personalizadas

### 5. **Re-entrenamiento** (`/airflow/dags/retrain_the_model.py`)
- **Tecnología**: MLflow, Airflow
- **Funcionalidades**:
  - Entrenamiento automático de modelos challenger
  - Comparación con modelo champion
  - Promoción automática del mejor modelo
  - Versionado de modelos

<br />

## Flujo de Trabajo

### 1. **Preparación de Datos**
```bash
# ETL Pipeline
Datos Raw → Limpieza → Imputación → Dummy Variables → Normalización → Train/Test Split
```

### 2. **Experimentación y Entrenamiento**
```bash
# Jupyter Notebook + Optuna
Datos → Optimización → Múltiples Modelos → Validación → Mejor Modelo → MLflow
```

### 3. **Deployment y Predicción**
```bash
# API + Frontend
Modelo Champion → FastAPI → Frontend → Usuario → Predicción
```

### 4. **Re-entrenamiento Continuo**
```bash
# Airflow DAG
Nuevos Datos → Challenger Model → Evaluación → Champion Update (si mejor)
```

<br />

## Tecnologías Utilizadas

| Componente | Tecnología | Versión |
|------------|------------|---------|
| **Orquestación** | Apache Airflow | 2.8+ |
| **ML Tracking** | MLflow | 2.10+ |
| **API Backend** | FastAPI | 0.104+ |
| **Frontend** | HTML5 + JavaScript | ES6+ |
| **Servidor Web** | Nginx | Alpine |
| **Base de Datos** | PostgreSQL | 13+ |
| **Almacenamiento** | MinIO (S3) | Latest |
| **Containerización** | Docker + Compose | 20.10+ |
| **ML Framework** | Scikit-learn, PyCaret | Latest |
| **Optimización** | Optuna | 3.0+ |

<br />

## Uso del Sistema

### Para Usuarios Finales
1. **Acceder al Frontend**: http://localhost:3000
2. **Completar formulario** con datos del paciente
3. **Obtener predicción** instantánea
4. **Interpretar resultados** (riesgo bajo/alto)

### Para Data Scientists
1. **Experimentar** en notebooks (`/notebook_example/`)
2. **Ejecutar pipelines** en Airflow (http://localhost:8080)
3. **Monitorear experimentos** en MLflow (http://localhost:5000)
4. **Analizar métricas** y comparar modelos

### Para DevOps/MLOps
1. **Gestionar contenedores** con Docker Compose
2. **Monitorear logs** de cada servicio
3. **Configurar variables** de entorno
4. **Escalar servicios** según demanda

<br />

## Configuración Avanzada

### Variables de Entorno
```bash
# Crear archivo .env
FRONTEND_PORT=3000
FASTAPI_PORT=8800
MLFLOW_PORT=5000
AIRFLOW_PORT=8080
```

### Desarrollo Local
```bash
# Solo frontend y API
docker compose up frontend fastapi mlflow postgres s3 -d

# Solo pipelines
docker compose --profile airflow up -d
```

<br />

## Documentación Adicional

- **[INSTRUCCIONES_FRONTEND.md](INSTRUCCIONES_FRONTEND.md)**: Guía completa del frontend
- **[frontend/README.md](frontend/README.md)**: Documentación técnica del frontend
- **Swagger API**: http://localhost:8800/docs (cuando esté ejecutándose)

<br />

## Estado del Proyecto

### Completado
- ✅ Pipeline ETL completo con Airflow
- ✅ Entrenamiento y registro de modelos con MLflow
- ✅ API FastAPI con predicciones
- ✅ **Frontend web interactivo**
- ✅ Sistema Champion/Challenger
- ✅ Re-entrenamiento automático
- ✅ Containerización completa
- ✅ Documentación exhaustiva



<br />

## Contribución

Este proyecto está basado en la solución propuesta en la materia:
- [Ejemplo de Implementación de Heart Disease](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation)

Todos los componentes han sido modificados y refactorizados para el caso de uso específico de predicción de ACV, con mejoras significativas en la arquitectura y nuevos componentes como el frontend web.

<br />

---

*💡 Para comenzar rápidamente en linux, ejecuta `./iniciar_sistema.sh` y abre http://localhost:3000*
