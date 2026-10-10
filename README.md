# TC5061_Experimentacion_MLFlow_VazquezAndres

Actividad Individual - Código de experimentación base con MLFlow.

Se continúa trabajando con el modelo base de la semana anterior (Semana 3) y se mantiene en este proyecto en la carpeta notebooks como referencia. Cookiecutter Data Science (CCDS) no se menciona en las instrucciones ni en la rúbrica de la actividad, pero generé la estructura del proyecto con esa herramienta (`ccds`) como práctica recomendada en la sesión síncrona 2. El modelo base de la semana pasada fue de regresión lineal múltiple para predecir `charges` en `data/raw/insurance.csv`. Se crea y trabaja en un script de entrenamiento parametrizado, `medical_cost/modeling/train.py`. Cada corrida registra en MLflow sus hiperparámetros, las métricas (RMSE, MAE y R2), un reporte de métricas y una gráfica de predicciones como artefactos y el modelo entrenado, todo dentro del experimento `Medical_Cost_Experimentacion_LR`.

El notebook de `notebooks/` se conserva solo para la exploración de los datos y referencia.

## Estructura

```
data/raw/insurance.csv          : dataset Medical Cost Personal Datasets, no se modifica, es el raw
medical_cost/                   : código del proyecto
    config.py                   : rutas y configuración de MLflow
    dataset.py                  : carga y limpieza de los datos
    features.py                 : preprocesamiento
    plots.py                    : gráfica de predicciones
    modeling/
        evaluate.py             : métricas
        train.py                : argumentos, entrenamiento y registro en MLflow
notebooks/                      : notebook del modelo base (Semana 3), solo referencia
reports/                        : reportes y figuras que generan las corridas
MLproject, python_env.yaml      : ejecución con MLflow Projects
Makefile, pyproject.toml, requirements.txt
```

## Setup

```bash
python3.13 -m venv venv3.13
source venv3.13/bin/activate
python -m pip install -r requirements.txt
```

El proyecto se desarrolló con Python 3.13.9.

## Levantar la MLflow UI

En una terminal aparte, con el entorno activado y desde la raíz del proyecto:

```bash
mlflow ui --port 5001
```

UI: http://localhost:5001 (uso el puerto 5001 porque en Mac el 5000 lo ocupa AirPlay Receiver).

## Ejecutar el script

Los comandos se ejecutan desde la raíz del proyecto y con `python -m`, porque `medical_cost` es un paquete (no sirve `python medical_cost/modeling/train.py`).

Con los valores por defecto:

```bash
python -m medical_cost.modeling.train
```

Cambiando los hiperparámetros desde la línea de comandos:

```bash
python -m medical_cost.modeling.train --fit_intercept 1 --positive 1 --tol 0.01 --seed 42
```

Guía de argumentos con argparse usados en `python -m medical_cost.modeling.train`:

- `--fit_intercept`, Default: 1, Descripción: 1 ajusta el intercepto, 0 lo fuerza a cero.
- `--positive`, Default: 0, Descripción: 1 obliga a que los coeficientes sean positivos, 0 no.
- `--tol`, Default: 1e-6, Descripción: tolerancia del solver de mínimos cuadrados, se aplica con `--positive 0`; con `--positive 1` no tuvo efecto en las pruebas.
- `--seed`, Default: 42, Descripción: semilla del split train/test.

Con los valores por defecto se obtiene la línea base del notebook (RMSE 5956.34, MAE 4177.05).

También se puede ejecutar como MLflow Project (con la UI levantada en el puerto 5001):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
mlflow run . --env-manager local --experiment-name Medical_Cost_Experimentacion_LR -P positive=1 -P seed=7
```

Los parámetros del `MLproject` son `fit_intercept`, `positive`, `tol` y `seed`. La opción `--env-manager local` usa el entorno que ya está activo, por eso primero se instalan las dependencias de `requirements.txt`.

## Reproducibilidad

La semilla (`--seed`) se usa en `np.random.seed` y en `random_state` de `train_test_split`. Dos corridas con los mismos argumentos dan las mismas métricas. Las versiones de las librerías están fijadas en `requirements.txt`.

En mi experimento las corridas 1 y 8 usan los mismos argumentos (los valores por defecto) a propósito, para comprobar la reproducibilidad. Los reportes y las gráficas de `reports/` llevan en su nombre los argumentos de la corrida, así que la corrida 8 sobrescribe en esa carpeta los archivos de la 1; por eso ahí hay 8 archivos de cada tipo y no 9. En MLflow no se pierde nada: cada corrida guarda su propia copia como artefacto y las dos aparecen por separado en la UI (mismo nombre, distinto Run ID). Su comparación directa no muestra diferencias en parámetros ni en métricas (Figuras 14 y 15 del PDF). Los archivos de reports/ se suben al repositorio únicamente como parte de la evaluación académica de esta actividad, ya que parte de ella es evaluar mis corridas de experimentación. En un proyecto real se generarían en cada corrida y no se versionarían, porque ya quedan registrados como artefactos en MLflow.

## Decisiones y justificación

La rúbrica y las instrucciones de la actividad no piden varias de las cosas de este proyecto. Las agregué porque en la sesión síncrona 2 del 8 de octubre de 2026 se vieron y recomendaron como buenas prácticas, y las dejo justificadas aquí:

- **Cookiecutter Data Science (CCDS).** No se menciona en las instrucciones. Generé la estructura con `ccds` (versión 2.3.0) como práctica para las siguientes entregas, porque en la sesión 2 se recomendó y se indicó que será obligatorio más adelante.
- **Más de un archivo .py.** Además de `train.py` hay `config.py`, `dataset.py`, `features.py`, `plots.py` y `evaluate.py`. En la sesión 2 se recomendó separar la carga de datos, el preprocesamiento, el entrenamiento y las métricas, con un script principal que invoque las funciones. Eso también cubre la parte de la rúbrica de funciones con responsabilidades claras.
- **Notebook en `notebooks/`.** Conservé sin cambios en su contenido el notebook del modelo base de la semana 3 (solo agregué "(Semana 3)" al título) como punto de partida y referencia. El código de entrenamiento de esta actividad vive en `medical_cost/`; el notebook queda como referencia.
- **MLproject y python_env.yaml.** No los piden las instrucciones. Los agregué porque en la sesión 2 se vio MLflow Projects para ejecutar el proyecto con un solo comando y parámetros tipados. La ejecución directa con `python -m` se mantiene.
- **Gráfica de predicciones.** Las instrucciones de la actividad piden al menos un artefacto y dan como ejemplo una curva. Además del reporte de métricas, registro como artefacto la gráfica de reales contra predichos del notebook, guardada en `reports/figures/`.
- **Configuración fuera del código.** Las rutas y la URI de MLflow están en `config.py` y la URI se puede cambiar con la variable `MLFLOW_TRACKING_URI`, siguiendo la recomendación de la sesión 2 de separar la configuración.
- **Programación orientada a objetos.** Era opcional esta semana según la indicación del profesor durante el chat de la sesión 2, pero la apliqué con una clase, `ModelRun`, que guarda la configuración, el modelo y las métricas de cada corrida, siguiendo el ejemplo visto en la sesión 2. El resto son funciones con comportamientos específicos.
- **Registros de MLflow.** `mlruns/` y `mlflow.db` no se versionan. En la sesión 2, la diapositiva de lo que va y no va en Git deja fuera `mlruns/`, `data/`, `models/` y `.env`, y `mlflow.db` es el mismo tipo de archivo, generado por MLflow en cada ejecución. La evidencia de las corridas son las capturas del PDF, como piden las instrucciones.
- **Dependencias.** Las versiones están fijadas en `requirements.txt` (lo piden las instrucciones) y en `python_env.yaml` (MLflow Projects, como práctica de la sesión 2).
- **Dataset en el repositorio.** Sé que en la sesión 2 se mostró `data/` fuera de Git en una de las diapositivas "What goes in Git (and what doesn't)", pensado para datos grandes, y que en un proyecto real también conviene dejarlo fuera porque los datos pueden ser sensibles. Aun así, en esta ocasión versiono `data/raw/insurance.csv` por tres razones. Primera, la actividad dice textualmente "cualquier persona pueda ejecutar tu script desde la línea de comandos" y sin el archivo el script no corre a partir del repositorio. Segunda, la reproducibilidad vale 10% de la rúbrica. Tercera, es una actividad académica con un dataset público de Kaggle (Medical Cost Personal Datasets) de unos 54 KB. En un proyecto real lo dejaría fuera de Git y lo descargaría desde su fuente (Kaggle) o desde un almacenamiento en la nube a `data/raw/`.

Las demás carpetas (`models`, `references` y `docs`) son las que genera `ccds` y quedan vacías. `ccds` también genera `data/interim`, `data/processed` y `data/external`, pero no se suben a Git porque `data/` está fuera del repositorio (salvo por lo antes mencionado: `data/raw/insurance.csv`).