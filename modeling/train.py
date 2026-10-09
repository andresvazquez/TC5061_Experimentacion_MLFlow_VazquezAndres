# Importamos las librerias que se usan
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Configuracion fija del experimento
TRACKING_URI = "http://localhost:5001"  # Se usa 5001 porque en mi Mac el puerto 5000 es utilizado por AirPlay
EXPERIMENTO = "Medical_Cost_Experimentacion_LR"
RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "raw" / "insurance.csv"
CAT_FEATURES = ['sex', 'smoker', 'region']
NUM_FEATURES = ['age', 'bmi', 'children']
TEST_SIZE = 0.2


def leer_argumentos():
    # Argumentos de linea de comandos, todos con valor por defecto
    parser = argparse.ArgumentParser(description="Entrena la regresion lineal base sobre insurance.csv y registra la corrida en MLflow")
    parser.add_argument("--fit_intercept", type=int, choices=[0, 1], default=1, help="1 ajusta el intercepto, 0 lo fuerza a cero (default: 1)")
    parser.add_argument("--positive", type=int, choices=[0, 1], default=0, help="1 obliga a que los coeficientes sean positivos, 0 no (default: 0)")
    parser.add_argument("--tol", type=float, default=1e-6, help="tolerancia del ajuste, solo se usa con --positive 1 (default: 1e-6)")
    parser.add_argument("--seed", type=int, default=42, help="semilla del split train/test (default: 42)")
    return parser.parse_args()


def cargar_datos(ruta):
    # Se elimina el duplicado y se reinicia el índice
    df = pd.read_csv(ruta)
    df = df.drop_duplicates().reset_index(drop=True)

    # Variables de entrada (X) y variable de salida/target (y)
    X = df.drop(columns='charges')
    y = df['charges']
    return X, y


def construir_modelo(args):
    # Preprocesamiento: one-hot para las categóricas (drop='first' para evitar la colinealidad) y las numéricas pasan sin transformación
    # Los imputers solo actuarían si algún día hubiera valores faltantes
    preprocessing = ColumnTransformer(transformers=[("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                                                            ("onehot", OneHotEncoder(drop="first"))]), CAT_FEATURES),
                                                    ("num", SimpleImputer(strategy="median"), NUM_FEATURES)])

    # Pipeline con el preprocesamiento y el modelo base
    return Pipeline(steps=[('preprocessing', preprocessing),
                           ('regressor', LinearRegression(fit_intercept=bool(args.fit_intercept),
                                                          positive=bool(args.positive),
                                                          tol=args.tol))])


def evaluar(model, X_test, y_test):
    # Predicciones sobre el conjunto de prueba y cálculo de las métricas
    y_pred = model.predict(X_test)
    return {"rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)}


def guardar_reporte(args, metricas):
    # Reporte de la configuración y las métricas de la corrida en un archivo de texto
    reportes = Path("reports")
    reportes.mkdir(exist_ok=True)
    reporte = reportes / f"reporte_regresion_s{args.seed}_int{args.fit_intercept}_pos{args.positive}_tol{args.tol}.txt"
    reporte.write_text(f"Modelo: LinearRegression\n"
                       f"fit_intercept: {bool(args.fit_intercept)}\n"
                       f"positive: {bool(args.positive)}\n"
                       f"tol: {args.tol}\n"
                       f"seed: {args.seed}\n"
                       f"RMSE: {metricas['rmse']:.2f}\n"
                       f"MAE: {metricas['mae']:.2f}\n"
                       f"R2: {metricas['r2']:.4f}\n")
    return reporte


def registrar_corrida(model, args, metricas, X_train, X_test):
    # Hiperparámetros y configuración de la corrida
    mlflow.log_param("fit_intercept", bool(args.fit_intercept))
    mlflow.log_param("positive", bool(args.positive))
    mlflow.log_param("tol", args.tol)
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("n_train", X_train.shape[0])
    mlflow.log_param("n_test", X_test.shape[0])

    # Métricas
    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)

    # Artefacto: reporte de las métricas en un archivo de texto
    mlflow.log_artifact(str(guardar_reporte(args, metricas)))

    # Modelo con su firma y un ejemplo de entrada
    # Se hace una firma de esquema manual declarada para conservar age y children como enteros (evita el warning de esquema inferido)
    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "age"), ColSpec("string", "sex"), ColSpec("double", "bmi"),
                       ColSpec("long", "children"), ColSpec("string", "smoker"), ColSpec("string", "region")]),
        outputs=Schema([ColSpec("double")]))
    mlflow.sklearn.log_model(sk_model=model, name="linear_regression_model",
                             signature=signature, input_example=X_train.head(3),
                             serialization_format="cloudpickle")


def main():
    args = leer_argumentos()

    # Semilla fija para que dos corridas con los mismos argumentos den lo mismo
    np.random.seed(args.seed)

    X, y = cargar_datos(RUTA_DATOS)

    # División 80% entrenamiento y 20% prueba, con semilla para que sea reproducible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=args.seed)

    # Conexión con el servidor de tracking y selección (o creación) del experimento
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)

    run_name = f"lr_int{args.fit_intercept}_pos{args.positive}_tol{args.tol}_s{args.seed}"
    with mlflow.start_run(run_name=run_name):
        model = construir_modelo(args)

        # Entrenamos solo con los datos de entrenamiento
        model.fit(X_train, y_train)

        metricas = evaluar(model, X_test, y_test)
        registrar_corrida(model, args, metricas, X_train, X_test)

    print(run_name)
    print('RMSE:', round(metricas['rmse'], 2))
    print('MAE:', round(metricas['mae'], 2))
    print('R2:', round(metricas['r2'], 4))


if __name__ == "__main__":
    main()