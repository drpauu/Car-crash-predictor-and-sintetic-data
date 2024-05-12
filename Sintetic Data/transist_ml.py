import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import networkx as nx
import matplotlib.pyplot as plt

def generate_data(n_routes):
    """ Genera dades simulades per a rutes i trànsit. """
    np.random.seed(42)
    origins = np.random.choice(['CityA', 'CityB', 'CityC'], n_routes)
    destinations = np.random.choice(['CityA', 'CityB', 'CityC'], n_routes)
    distances = np.random.uniform(5, 100, n_routes)
    times = distances / np.random.uniform(30, 70, n_routes) * 60  # Temps en minuts
    traffic_levels = np.random.choice(['Low', 'Medium', 'High'], n_routes)
    
    data = pd.DataFrame({
        'origin': origins,
        'destination': destinations,
        'distance': distances,
        'time': times,
        'traffic': traffic_levels
    })
    return data

def preprocess_data(df):
    """ Preprocessa les dades: codifica categories i normalitza distàncies. """
    df['traffic'] = df['traffic'].map({'Low': 0, 'Medium': 1, 'High': 2})
    scaler = StandardScaler()
    df['distance'] = scaler.fit_transform(df[['distance']])
    return df

def train_model(data):
    """ Entrena un model de regressió lineal per predir el temps basat en distància i trànsit. """
    X = data[['distance', 'traffic']]
    y = data['time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model MSE: {mse:.2f}')
    return model

def create_graph(data):
    """ Crea un graf de rutes utilitzant NetworkX. """
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row['origin'], row['destination'], weight=row['time'])
    return G

def optimize_routes(graph, model, df):
    """ Optimitza les rutes ajustant els pesos basats en el model de ML. """
    for u, v, d in graph.edges(data=True):
        route_data = df[(df['origin'] == u) & (df['destination'] == v)]
        if not route_data.empty:
            predicted_time = model.predict([[route_data.iloc[0]['distance'], route_data.iloc[0]['traffic']]])
            graph[u][v]['weight'] = predicted_time
    return graph

def find_optimal_route(graph, origin, destination):
    """ Troba la ruta òptima utilitzant el camí mínim. """
    path = nx.shortest_path(graph, source=origin, target=destination, weight='weight')
    return path

# Flux principal del script
n_routes = 100
data = generate_data(n_routes)
data = preprocess_data(data)
model = train_model(data)
graph = create_graph(data)
graph = optimize_routes(graph, model, data)

# Visualització de la ruta òptima
origin, destination = 'CityA', 'CityB'
optimal_route = find_optimal_route(graph, origin, destination)
print(f'Ruta òptima de {origin} a {destination}: {optimal_route}')

# Opcional: Visualitza el graf
nx.draw_networkx(graph, with_labels=True)
plt.show()
