import argparse

import mlflow
from mlflow.models import ModelSignature
import mlflow.sklearn
from mlflow.types import ColSpec, Schema
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from medical_cost.config import (
    EXPERIMENTO,
    FIGURES_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TRACKING_URI,
)
from medical_cost.dataset import cargar_datos
from medical_cost.features import construir_preprocesamiento
from medical_cost.modeling.evaluate import evaluar
from medical_cost.plots import graficar_predicciones

RUTA_DATOS = RAW_DATA_DIR / "insurance.csv"
TEST_SIZE = 0.2


def leer_argumentos():
    # Argumentos de linea de comandos, todos con valor por defecto
    parser = argparse.ArgumentParser(
        description="Entrena la regresion lineal base sobre insurance.csv y registra la corrida en MLflow"
    )
    parser.add_argument(
        "--fit_intercept",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 ajusta el intercepto, 0 lo fuerza a cero (default: 1)",
    )
    parser.add_argument(
        "--positive",
        type=int,
        choices=[0, 1],
        default=0,
        help="1 obliga a que los coeficientes sean positivos, 0 no (default: 0)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="tolerancia del solver de minimos cuadrados, se aplica con --positive 0 (default: 1e-6)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="semilla del split train/test (default: 42)"
    )
    return parser.parse_args()


def construir_modelo(args):
    # Pipeline con el preprocesamiento y el modelo base
    return Pipeline(
        steps=[
            ("preprocessing", construir_preprocesamiento()),
            (
                "regressor",
                LinearRegression(
                    fit_intercept=bool(args.fit_intercept),
                    positive=bool(args.positive),
                    tol=args.tol,
                ),
            ),
        ]
    )


class ModelRun:
    # Guarda la configuración, el modelo y las métricas de una corrida
    def __init__(self, args):
        self.args = args
        self.nombre = f"lr_int{args.fit_intercept}_pos{args.positive}_tol{args.tol}_s{args.seed}"
        self.model = construir_modelo(args)
        self.metricas = {}
        self.y_pred = None

    def entrenar(self, X_train, y_train):
        # Entrenamos solo con los datos de entrenamiento
        self.model.fit(X_train, y_train)

    def evaluar(self, X_test, y_test):
        self.metricas, self.y_pred = evaluar(self.model, X_test, y_test)

    def guardar_reporte(self):
        # Reporte de la configuración y las métricas de la corrida en un archivo de texto
        reporte = REPORTS_DIR / f"reporte_regresion_{self.nombre}.txt"
        reporte.write_text(
            f"Modelo: LinearRegression\n"
            f"fit_intercept: {bool(self.args.fit_intercept)}\n"
            f"positive: {bool(self.args.positive)}\n"
            f"tol: {self.args.tol}\n"
            f"seed: {self.args.seed}\n"
            f"RMSE: {self.metricas['rmse']:.2f}\n"
            f"MAE: {self.metricas['mae']:.2f}\n"
            f"R2: {self.metricas['r2']:.4f}\n"
        )
        return reporte

    def registrar(self, X_train, X_test, reporte, figura):
        # Hiperparámetros y configuración de la corrida
        # Se registran tal cual llegan (0/1) para que coincidan con lo que guarda mlflow run con -P
        mlflow.log_param("fit_intercept", self.args.fit_intercept)
        mlflow.log_param("positive", self.args.positive)
        mlflow.log_param("tol", self.args.tol)
        mlflow.log_param("seed", self.args.seed)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])

        # Métricas
        for nombre, valor in self.metricas.items():
            mlflow.log_metric(nombre, valor)

        # Artefactos: reporte de las métricas y gráfica de predicciones
        mlflow.log_artifact(str(reporte))
        mlflow.log_artifact(str(figura))

        # Modelo con su firma y un ejemplo de entrada
        # Se hace una firma de esquema manual declarada para conservar age y children como enteros (evita el warning de esquema inferido)
        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("long", "age"),
                    ColSpec("string", "sex"),
                    ColSpec("double", "bmi"),
                    ColSpec("long", "children"),
                    ColSpec("string", "smoker"),
                    ColSpec("string", "region"),
                ]
            ),
            outputs=Schema([ColSpec("double")]),
        )
        mlflow.sklearn.log_model(
            sk_model=self.model,
            name="linear_regression_model",
            signature=signature,
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )


def main():
    args = leer_argumentos()

    # Semilla fija para que dos corridas con los mismos argumentos den lo mismo
    np.random.seed(args.seed)

    X, y = cargar_datos(RUTA_DATOS)

    # División 80% entrenamiento y 20% prueba, con semilla para que sea reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed
    )

    # Conexión con el servidor de tracking y selección (o creación) del experimento
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)

    corrida = ModelRun(args)
    with mlflow.start_run(run_name=corrida.nombre):
        corrida.entrenar(X_train, y_train)
        corrida.evaluar(X_test, y_test)

        figura = FIGURES_DIR / f"predicciones_{corrida.nombre}.png"
        graficar_predicciones(y_test, corrida.y_pred, figura)
        reporte = corrida.guardar_reporte()
        corrida.registrar(X_train, X_test, reporte, figura)

    print(corrida.nombre)
    print("RMSE:", round(corrida.metricas["rmse"], 2))
    print("MAE:", round(corrida.metricas["mae"], 2))
    print("R2:", round(corrida.metricas["r2"], 4))


if __name__ == "__main__":
    main()
