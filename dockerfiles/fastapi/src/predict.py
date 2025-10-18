import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

def run(features_dict, data_dict, model, debug = False):
  # 1. Crear el DataFrame directamente desde el diccionario
  # Esto preserva los dtypes (int, float, str) originales
  features_df = pd.DataFrame(features_dict, index=[0])

  # 2. Procesar columnas categóricas
  for categorical_col in data_dict["categorical_columns"]:
    features_df[categorical_col] = features_df[categorical_col].astype(str)
    categories = data_dict["categories_values_per_categorical"][categorical_col]
    features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)

  if (debug):
    print("\n--- 1. Después de forzar tipos categóricos ---")
    features_df.info()

  # 3. Convertir categóricas a dummies
  features_df = pd.get_dummies(data=features_df,
                               columns=data_dict["categorical_columns"],
                               drop_first=True)

  if (debug):
    print("\n--- 2. Después de get_dummies ---")
    features_df.info()

  # 4. Reordenar columnas
  # Asegurarse de que las columnas numéricas que no eran dummies (age, bmi, etc.)
  # estén presentes en data_dict["columns_after_dummy"]
  features_df = features_df.reindex(columns=data_dict["columns_after_dummy"], fill_value=0)

  if (debug):
    print("\n--- 3. Después de reindexar ---")
    print(features_df.head(1))

  # 5. Escalar los datos
  # Esta operación ahora funcionará porque las columnas son numéricas
  features_df = (features_df - data_dict["standard_scaler_mean"]) / data_dict["standard_scaler_std"]

  # 6. Hacer la predicción
  prediction = model.predict(features_df)

  if (debug):
    print(f"\nPredicción final: {prediction}")

  return prediction