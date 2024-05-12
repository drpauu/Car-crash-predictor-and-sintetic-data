import pandas as pd
from sklearn.preprocessing import StandardScaler

def collect_and_prepare_data(filename):
    """
    Funció per recollir i preparar les dades per a entrenament de models de ML.

    Args:
    filename (str): Ruta al fitxer que conté les dades històriques.

    Returns:
    DataFrame: Dades processades i netejades.
    """
    # Carregar dades des d'un fitxer CSV
    data = pd.read_csv(filename)

    # Exemple de preprocessament de dades
    # Suposem que el fitxer conté les columnes 'origin', 'destination', 'distance', 'travel_time', 'safety'
    # Netejar dades faltants
    data.dropna(inplace=True)

    # Convertir dades categòriques a numèriques si és necessari
    # Per exemple, si 'origin' i 'destination' són categòriques
    data['origin'] = data['origin'].astype('category').cat.codes
    data['destination'] = data['destination'].astype('category').cat.codes

    # Normalitzar les dades numèriques
    scaler = StandardScaler()
    data[['distance', 'travel_time', 'safety']] = scaler.fit_transform(data[['distance', 'travel_time', 'safety']])

    return data

if __name__ == '__main__':
    # Inicia l'execució del programa principal si s'executa com a script
    # Suposant que les dades estan en un fitxer anomenat 'route_data.csv'
    processed_data = collect_and_prepare_data('datos.txt')
    main()  
