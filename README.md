# Quinta lista de ejercicios – Introducción al Machine Learning

## Contenido

- `data/Datos_Hitters.csv`: base de datos suministrada.
- `notebooks/Quinta_lista_ML_Hitters.ipynb`: desarrollo reproducible de los cuatro ejercicios.
- `src/analisis.py`: funciones reutilizables para riesgo cuadrático, ajuste de modelos, selección hacia adelante y pruebas t.
- `results/`: tablas y gráficas generadas al ejecutar el cuaderno.
- `requirements.txt`: dependencias.

## Cómo ejecutar

1. Descargue el repositorio.
2. Abra una terminal en la carpeta raíz del proyecto.
3. Cree un entorno virtual (opcional, recomendado):

   ```bash
   python -m venv .venv
   ```

4. Active el entorno:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

5. Instale dependencias:

   ```bash
   pip install -r requirements.txt
   ```

6. Inicie Jupyter:

   ```bash
   jupyter notebook
   ```

7. Abra `notebooks/Quinta_lista_ML_Hitters.ipynb` y ejecute **Restart Kernel and Run All**.

El cuaderno localiza el directorio del proyecto tanto si se ejecuta desde la raíz como desde `notebooks/`.

## Decisiones metodológicas

- La columna `index` se excluye porque identifica filas y no es una variable predictora.
- `Salary` es la variable respuesta.
- Las 19 columnas restantes son las predictoras.
- La partición se realiza en dos etapas con `random_state=31`: 120 para desarrollo y 143 para prueba; luego 50 para entrenamiento y 70 para validación.
- Los ejercicios 1–3 usan exclusivamente entrenamiento y validación.
- La prueba se utiliza únicamente en el ejercicio 4.
- Todos los modelos son regresiones lineales con intercepto.
- El riesgo solicitado se calcula como el promedio de los errores al cuadrado (MSE).

## Resultados de referencia

Los riesgos pueden variar solo si se modifica la partición, las columnas o el procedimiento. Con el archivo suministrado y las instrucciones de la guía:

| Modelo | Variables | Riesgo entrenamiento | Riesgo validación | Riesgo prueba |
|---|---:|---:|---:|---:|
| Todas las variables | 19 | 68584.82 | 320423.19 | 394308.69 |
| Mayor correlación | 7 | 115487.25 | 133035.61 | 145923.29 |
| Pruebas t | 3 | 161221.36 | 140085.19 | 158772.47 |
| Selección hacia adelante | 3 | 126982.32 | 130618.20 | 126939.60 |

El modelo elegido mediante validación es selección hacia adelante, con `CRBI`, `Years` y `CRuns`. También obtiene el menor riesgo de prueba entre los cuatro modelos comparados. Esto último es un resultado observado en esta partición, no una garantía general de que la validación siempre seleccione el mejor modelo en prueba.
