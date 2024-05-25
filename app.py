from flask import Flask, request, jsonify
from pymongo import MongoClient
import json
import pymongo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
# Ensemble algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, ConfusionMatrixDisplay, recall_score, roc_auc_score
# Create a pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To measure performance
from sklearn import metrics

# Save the model
import joblib

app = Flask(__name__)

#Pruebas en docker mongo
client = MongoClient('mongodb://172.17.0.2:27017/')

#pruebas en local
#client = MongoClient('mongodb://localhost:27017/')

db = client['challenge']  # Nombre de tu base de datos
collection = db['challenge']
data = []

@app.route('/')
def index():
    return '/Consulta : requiere un Id para consultar el puntaje! <br>, /Agregar : Requiere un Json con los datos para calcular el porcentaje <br>, /ML : sirve para ejecutar y entrenar el servicio'

# Método GET para obtener datos por medio de un ID de Empleado
@app.route('/Consulta', methods=['GET'])
def get_data():
    employee_number = request.args.get('EmployeeNumber')
    employee  = collection.find_one({"EmployeeNumber": int(employee_number)})
    employee['_id'] = str(employee['_id'])
    # Crea un diccionario con el número de empleado y su puntuación
    employee_data = {
        "employee_number": employee["EmployeeNumber"],
        "score": employee["turnover_score"]
    }
    return jsonify(employee_data)

# Método POST para agregar datos
@app.route('/Agregar', methods=['POST'])
def hr_prediccion():
    content = request.json
    clf = joblib.load('clf.zahoree')

    df = pd.DataFrame([content])
    ID = df['EmployeeNumber'][0]
    df.drop(columns=['EmployeeNumber'], inplace=True)
    prediction = clf.predict_proba(df)  #
    insert = {'EmployeeNumber': int(ID), 'turnover_score': list(prediction[:, 1])[0]}
    collection.insert_one(insert)

    return 'Agregado Correctamente!'

# Método POST para ML
@app.route('/ML', methods=['GET'])
def  Machine():
    np.random.seed(2024)
    hrdata = pd.read_csv('./HR_Employee_Attrition.csv')
    hrdata.head()
    hrdata.shape
    train_hrdata = hrdata.drop(columns=['EmployeeCount', 'EmployeeNumber', 'JobLevel',
                                        'Over18', 'StandardHours', 'TotalWorkingYears'])
    train_hrdata['Attrition'] = train_hrdata.Attrition.map({'Yes': 1,
                                                            'No': 0})
    categorical_attributes = ['BusinessTravel', 'OverTime',
                              'Department', 'EducationField',
                              'Gender', 'JobRole', 'MaritalStatus']

    rf = RandomForestClassifier(max_depth=10,
                                max_features=12,  #
                                n_estimators=180,  #
                                random_state=2024,
                                n_jobs=-1)

    cat_pipe = ColumnTransformer([('ordinal_encoder', OrdinalEncoder(), categorical_attributes)],
                                 remainder='passthrough')

    pipe_model = Pipeline([
        ('encoder', cat_pipe),
        ('classification', rf)
    ])
    df1 = train_hrdata[train_hrdata.Attrition == 0].sample(600).reset_index(drop=True)
    df2 = train_hrdata[train_hrdata.Attrition == 1]
    train_set = pd.concat([df1, df2, df2], axis=0).reset_index(drop=True)
    x = train_set.drop(columns=['Attrition'])  ### Drop before having the target variable
    y = train_set['Attrition']

   # print(x.shape)
   # print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=2024,
                                                        test_size=0.2,
                                                        stratify=y)
    pipe_model.fit(x_train, y_train)
    y_pred = pipe_model.predict(x_test)

    #print('Accuracy Score of Random Forest Classifier is: ', metrics.accuracy_score(y_test, y_pred))
    #print('Recall Score of Random Forest Classifier Model is: ', metrics.recall_score(y_test, y_pred))
   # print(metrics.classification_report(y_test, y_pred))


    val_cols = list(train_set.columns)
    val_cols.remove('Attrition')
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].set_title('Confusion Matrix:')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, cmap='Blues', ax=ax[0])
    ax[0].grid(False)

    scoring = pipe_model.predict_proba(x_test[val_cols])[:, 1]
    # Compute ROC metrics:
    fpr, tpr, thresholds = roc_curve(y_test.values, scoring)
    roc_auc = auc(fpr, tpr)

    ax[1].set_title('ROC Curve - Classifier')
    ax[1].plot(fpr, tpr, label='AUC = %0.2f' % roc_auc, c='teal')
    ax[1].plot([0, 1], [0, 1], '--', c='skyblue')
    ax[1].legend(loc='lower right')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('False Positive Rate')

    hrdata["turnover_score"] = pipe_model.predict_proba(hrdata[val_cols])[:, 1]  #
    hrdata[['EmployeeNumber', 'turnover_score']].head()
    # print (hrdata[['EmployeeNumber','turnover_score']].head())
    # hrdata[['EmployeeNumber','turnover_score']].to_csv('turnover_score_by_employee_number.csv', index=False)

    joblib.dump(pipe_model, 'clf.zahoree')
    clf = joblib.load('clf.zahoree')
    hrdata2 = pd.read_csv('./HR_Employee_Attrition.csv')
    collaborator_rn = np.random.choice(range(1, hrdata2.shape[1]))
    collaborator = pd.DataFrame(hrdata2.iloc[collaborator_rn, :]).T

    collaborator.drop(columns=['EmployeeCount',
                               'Attrition',
                               'JobLevel',
                               'Over18',
                               'StandardHours',
                               'TotalWorkingYears'], inplace=True)
    collaborator.to_json(orient="records")
    # Seleccionar una colección
    prueba = hrdata[['EmployeeNumber', 'turnover_score']].to_dict(orient='records')

    collection.insert_many(prueba)
    exito =   {"Datos insertados correctamente": "si"}
    return jsonify(exito)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')