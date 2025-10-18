import json
import pickle

from src import predict

print("Abriendo modelo")
with open('./files/model.pkl', 'rb') as file_ml:
    model_ml = pickle.load(file_ml)
print("Modelo abierto")

features_ninio = {
    "gender": "Male",
    "age": 1,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "No",
    "work_type": "children",
    "Residence_type": "Urban",
    "avg_glucose_level": 90.69,
    "bmi": 20.6,
    "smoking_status": "never smoked",
}

features_adulto_complicado = {
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


print("Cargando data de las columnas")
with open('./files/data.json', 'r') as file_data:
    data_dictionary = json.load(file_data)
print("Data de las columnas cargadas")

print("\n\n\nhaciendo predicciones (niño)")

predict.run(features_ninio, data_dictionary, model_ml, debug=True)

print("\n\n\nhaciendo predicciones (adulto complicado)")

predict.run(features_adulto_complicado, data_dictionary, model_ml, debug=True)
