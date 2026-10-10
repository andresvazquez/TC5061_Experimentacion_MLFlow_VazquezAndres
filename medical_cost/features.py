from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Columnas categóricas y numéricas
CAT_FEATURES = ["sex", "smoker", "region"]
NUM_FEATURES = ["age", "bmi", "children"]


def construir_preprocesamiento():
    # One-hot para las categóricas (drop='first' para evitar la colinealidad) y las numéricas pasan sin transformación
    # Los imputers solo actuarían si algún día hubiera valores faltantes: moda en categóricas y mediana en numéricas
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(drop="first")
    )
    return ColumnTransformer(
        [
            ("cat", cat_pipeline, CAT_FEATURES),
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ]
    )
