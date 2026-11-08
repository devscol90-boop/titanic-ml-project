# Proyecto de Machine Learning: Titanic 🚢

Este es mi primer proyecto de Ciencia de Datos. El objetivo es predecir qué pasajeros sobrevivieron al hundimiento del Titanic utilizando Python y Scikit-Learn.

## 📊 Tecnologías utilizadas
* **Python 3**
* **Pandas** (para la manipulación y limpieza de datos)
* **Scikit-Learn** (para el modelado predictivo)

## ⚙️ ¿Cómo funciona el modelo?
1. **Limpieza de datos:** Se rellenaron los valores nulos de la columna `Age` (edad) y `Fare` (tarifa) con la mediana. Se convirtieron las variables categóricas (Sexo) a valores numéricos.
2. **Modelo:** Se entrenó un clasificador **Random Forest** (`RandomForestClassifier`) con 100 árboles.
3. **Predicción:** Se generaron las predicciones para el conjunto de prueba y se exportaron en el archivo `submission.csv`.

## 🏅 Resultado
El modelo obtuvo un puntaje de precisión de **73.2%** en la tabla de posiciones de Kaggle.
