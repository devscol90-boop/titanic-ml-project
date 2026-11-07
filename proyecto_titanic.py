import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("1. Cargando datos...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# --- LIMPIEZA DE DATOS ---
print("2. Limpiando datos...")
# Rellenar edades vacías con la mediana
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

# Rellenar tarifas vacías
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Convertir Sexo a números (male=0, female=1)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# --- ENTRENAMIENTO ---
print("3. Entrenando el modelo...")
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X_train = train_data[features]
y_train = train_data['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- PREDICCIÓN ---
print("4. Generando predicciones para los datos de prueba...")
X_test = test_data[features]
predictions = model.predict(X_test)

# --- CREAR ARCHIVO DE ENVÍO ---
print("5. Guardando archivo 'submission.csv'...")
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("\n¡PROYECTO COMPLETADO CON EXITO! Archivo 'submission.csv' generado.")
