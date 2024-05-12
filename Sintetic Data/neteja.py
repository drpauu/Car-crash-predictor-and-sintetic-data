import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(filename):
    """
    Funció per preprocessar les dades de rutes.

    Args:
    filename (str): Ruta al fitxer de dades.

    Returns:
    X_train, X_test, y_train, y_test: Dades dividides en entrenament i prova.
    """
    # Carregar dades
    data = pd.read_csv(filename)
    
    # Eliminar files amb valors faltants en columnes crítiques
    data.dropna(subset=['travel_time', 'distance', 'safety'], inplace=True)
    
    # Definir transformadors per dades numèriques i categòriques
    numeric_features = ['distance', 'safety']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['origin', 'destination']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Crear el preprocessador utilitzant ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Preparar les característiques i la variable objectiu
    X = data.drop('travel_time', axis=1)
    y = data['travel_time']
    
    # Dividir les dades en conjunts d'entrenament i de prova
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Aplicar transformacions
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test
