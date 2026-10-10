import pandas as pd


def cargar_datos(ruta):
    # Se elimina el duplicado y se reinicia el índice
    df = pd.read_csv(ruta)
    df = df.drop_duplicates().reset_index(drop=True)

    # Variables de entrada (X) y variable de salida/target (y)
    X = df.drop(columns="charges")
    y = df["charges"]
    return X, y
