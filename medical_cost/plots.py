import matplotlib.pyplot as plt
import seaborn as sns


def graficar_predicciones(y_test, y_pred, ruta):
    # Valores reales contra predicciones y errores de predicción
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
    axes[0].set_xlabel("charges real")
    axes[0].set_ylabel("charges prediccion")
    sns.histplot(y_test - y_pred, ax=axes[1])
    axes[1].set_xlabel("error (real - prediccion)")
    fig.savefig(ruta)
    plt.close(fig)
