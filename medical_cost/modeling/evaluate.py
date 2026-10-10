from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluar(model, X_test, y_test):
    # Predicciones sobre el conjunto de prueba y cálculo de las métricas
    y_pred = model.predict(X_test)
    metricas = {
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    return metricas, y_pred
