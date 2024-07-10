import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, median_absolute_error

# Definimos la función get_metrics. Esta función recibirá cuatro argumentos:
# y_pred_test: son las predicciones del modelo para el conjunto de prueba (test set)
# y_test: son los valores verdaderos para el conjunto de prueba (test set)
# y_pred_train: son las predicciones del modelo para el conjunto de entrenamiento (train set)
# y_train: son los valores verdaderos para el conjunto de entrenamiento (train set)


def get_metrics(y_pred_test, y_test, y_pred_train, y_train):
  # Calculamos tres métricas para el conjunto de entrenamiento usando las predicciones y los valores verdaderos:
  # 1. R2 Score: es un número que indica qué tan bien las predicciones se acercan a los valores verdaderos.
  # 2. Median Absolute Error: es la mediana de los errores absolutos entre las predicciones y los valores verdaderos.
  # 3. Mean Absolute Percentage Error (MAPE): es el promedio del error porcentual entre las predicciones y los valores verdaderos.
  metrics_train = (
    r2_score(y_train, y_pred_train),
    median_absolute_error(y_train, y_pred_train),
    mean_absolute_percentage_error(y_train, y_pred_train) * 100  # Multiplicamos por 100 para obtener un porcentaje.
  )
  # Hacemos lo mismo para el conjunto de prueba.
  metrics_test = (
    r2_score(y_test, y_pred_test),
    median_absolute_error(y_test, y_pred_test),
    mean_absolute_percentage_error(y_test, y_pred_test) * 100  # Multiplicamos por 100 para obtener un porcentaje.
  )
  # Calculamos la diferencia entre las métricas del conjunto de prueba y entrenamiento.
  # Esto nos da una idea de cómo el modelo generaliza sobre datos no vistos durante el entrenamiento.
  metrics_diff = list(map(lambda x: x[1] - x[0], zip(metrics_train, metrics_test)))
  # Creamos un DataFrame de pandas para presentar los resultados de una manera legible.
  # El DataFrame tendrá tres filas (una para el conjunto de entrenamiento, una para el conjunto de prueba y una para la diferencia)
  # y tendrá tres columnas correspondientes a las métricas R2, Median AE y MAPE.
  results = pd.DataFrame(
    data=[metrics_train, metrics_test, metrics_diff],
    columns=['R2', 'Median AE', 'MAPE'],
    index=['Train set', 'Test set', 'Diferencia']
  )
  # La función retorna el DataFrame creado.
  return results
# Esta sería una forma de usar la función, pero necesitarías tener los datos reales de y_linear, y_test, yhat_train y y_train:
# my_results = get_metrics(y_linear_predict, y_test_real, yhat_train_predict, y_train_real)
# print(my_results)