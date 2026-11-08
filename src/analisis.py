"""Funciones reutilizables para la Quinta lista de Machine Learning."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def riesgo_cuadratico(y_real, y_estimado):
    """Calcula el promedio de los errores al cuadrado (MSE)."""
    y_real = np.asarray(y_real)
    y_estimado = np.asarray(y_estimado)
    return np.mean((y_real - y_estimado) ** 2)


def ajustar_y_evaluar(X_train, y_train, X_valid, y_valid, variables):
    """Ajusta regresión lineal con intercepto y calcula riesgos train/valid."""
    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(X_train[variables], y_train)
    riesgo_train = riesgo_cuadratico(y_train, modelo.predict(X_train[variables]))
    riesgo_valid = riesgo_cuadratico(y_valid, modelo.predict(X_valid[variables]))
    return modelo, riesgo_train, riesgo_valid


def seleccionar_hacia_adelante(X_train, y_train, X_valid, y_valid, variables):
    """
    Selección hacia adelante: en cada paso agrega la variable que minimiza
    el riesgo de entrenamiento. Nunca elimina variables ya seleccionadas.
    """
    seleccionadas = []
    restantes = list(variables)
    filas = []
    modelos = {}

    for numero in range(1, len(variables) + 1):
        candidatos = []
        for variable in restantes:
            conjunto = seleccionadas + [variable]
            modelo = LinearRegression(fit_intercept=True)
            modelo.fit(X_train[conjunto], y_train)
            r_train = riesgo_cuadratico(y_train, modelo.predict(X_train[conjunto]))
            r_valid = riesgo_cuadratico(y_valid, modelo.predict(X_valid[conjunto]))
            candidatos.append((r_train, variable, r_valid, modelo))

        # min conserva el orden original de variables en caso de empate.
        r_train, agregada, r_valid, modelo = min(candidatos, key=lambda x: x[0])
        seleccionadas.append(agregada)
        restantes.remove(agregada)
        modelos[numero] = modelo
        filas.append({
            "Número de variables": numero,
            "Variable agregada": agregada,
            "Riesgo entrenamiento": r_train,
            "Riesgo validación": r_valid,
            "Variables": ", ".join(seleccionadas)
        })

    tabla = pd.DataFrame(filas)
    mejor_numero = int(tabla["Riesgo validación"].idxmin()) + 1
    return tabla, mejor_numero, modelos[mejor_numero], seleccionadas[:mejor_numero]


def obtener_pvalores(X_train, y_train):
    """Obtiene los p-valores de las pruebas t de los coeficientes OLS."""
    ajuste = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    return ajuste, ajuste.pvalues.drop("const")
