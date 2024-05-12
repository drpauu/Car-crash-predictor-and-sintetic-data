from pymongo import MongoClient
import pandas as pd

def get_route_data(db_name, collection_name):
    """
    Funció per extreure dades de rutes des d'una col·lecció MongoDB.

    Args:
    db_name (str): Nom de la base de dades a MongoDB.
    collection_name (str): Nom de la col·lecció on es guarden les dades de les rutes.

    Returns:
    DataFrame: Un DataFrame de pandas amb les dades de les rutes.
    """
    # Connexió al servidor MongoDB (localhost per defecte)
    client = MongoClient('mongodb://localhost:27017/')

    # Accedir a la base de dades
    db = client[db_name]

    # Accedir a la col·lecció específica
    collection = db[collection_name]

    # Crear una consulta per recuperar totes les dades (ajustar segons necessitat)
    query = {}  # Un diccionari buit selecciona tots els documents

    # Recuperar les dades de la col·lecció
    data = list(collection.find(query))

    # Convertir la llista de dicionaris a un DataFrame de pandas
    df = pd.DataFrame(data)

    # Tancar la connexió MongoDB
    client.close()

    return df

# Exemple d'ús
db_name = 'routingDatabase'
collection_name = 'routes'
route_data = get_route_data(db_name, collection_name)
print(route_data.head())  # Mostrar les primeres files del DataFrame per verificació
