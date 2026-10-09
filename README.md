# TC5061_Experimentacion_MLFlow_VazquezAndres

Actividad Individual - Código de experimentación base con MLFlow.

Se continúa trabajando con el modelo base de la semana anterior (Semana 3) y se mantiene en este proyecto en la carpeta notebooks como referencia. No se hace mención de Cookiecutter Data Science (CCDS) en las instrucciones y rúbrica de la actividad, pero se hace un esfuerzo de acomodo semejante en carpetas de mismo nombre que la gestión de CCDS. El modelo base de la semana pasada fue de regresión lineal múltiple para predecir `charges` en `data/raw/insurance.csv`, se crea y trabaja en un script de entrenamiento parametrizado, `modeling/train.py`. Cada corrida registra en MLflow sus hiperparámetros, las métricas (RMSE, MAE y R2), un reporte de métricas como artefacto y el modelo entrenado, todo dentro del experimento `Medical_Cost_Experimentacion_LR`.

El notebook de `notebooks/` se conserva solo para la exploración de los datos y referencia.

## Estructura

```
data/raw/insurance.csv
modeling/train.py
notebooks/
requirements.txt
```

## Setup

```bash
virtualenv venv3.13 --python=python3.13.9
source venv3.13/bin/activate
pip install -r requirements.txt
```

## Levantar la MLflow UI

En una terminal aparte, con el entorno activado y desde la raíz del proyecto:

```bash
mlflow ui --port 5001
```

UI: http://localhost:5001 (uso el puerto 5001 porque en Mac el 5000 lo ocupa AirPlay Receiver).

## Ejecutar el script

Con los valores por defecto:

```bash
python modeling/train.py
```

Cambiando los hiperparámetros desde la línea de comandos:

```bash
python modeling/train.py --fit_intercept 1 --positive 1 --tol 0.01 --seed 42
```

Guía de argumentos con argparse usados en `python modeling/train.py`:

- `--fit_intercept`, Default: 1, Descripción: 1 ajusta el intercepto, 0 lo fuerza a cero.
- `--positive`, Default: 0, Descripción: 1 obliga a que los coeficientes sean positivos, 0 no.
- `--tol`, Default: 1e-6, Descripción: tolerancia del ajuste, solo se usa con `--positive 1`.
- `--seed`, Default: 42, Descripción: semilla del split train/test.

Con los valores por defecto se obtiene la línea base del notebook (RMSE 5956.34, MAE 4177.05).

## Reproducibilidad

La semilla (`--seed`) se usa en `np.random.seed` y en `random_state` de `train_test_split`. Dos corridas con los mismos argumentos dan las mismas métricas. Las versiones de las librerías están fijadas en `requirements.txt`.