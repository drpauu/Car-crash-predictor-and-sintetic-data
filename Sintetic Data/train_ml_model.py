from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_ml_model(data):
    """
    Funció per entrenar un model de machine learning utilitzant les dades proporcionades.

    Args:
    data (DataFrame): Dades preprocessades per a entrenament.

    Returns:
    model (object): Model entrenat.
    performance (dict): Diccionari amb mètriques de rendiment del model.
    """
    # Suposem que 'travel_time' és la variable objectiu i les altres són predictores
    X = data.drop('travel_time', axis=1)
    y = data['travel_time']

    # Dividir les dades en conjunts d'entrenament i de prova
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear i entrenar el model de regressió lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Avaluar el model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Retornar el model i les mètriques de rendiment
    performance = {'MSE': mse, 'R2 Score': r2}
    return model, performance
# Utilitzar les dades processades per entrenar el model
# porcessed data es crea amb la funció anterior
model, performance = train_ml_model(processed_data)

# Imprimir les mètriques de rendiment del model
print("Model Performance:")
print(f"MSE: {performance['MSE']}")
print(f"R2 Score: {performance['R2 Score']}")
