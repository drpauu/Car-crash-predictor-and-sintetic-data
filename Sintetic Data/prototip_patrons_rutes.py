import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import networkx as nx
import matplotlib.pyplot as plt

def generate_fake_data(num_samples=100):
    """ Genera dades fictícies per a les rutes i trànsit. """
    np.random.seed(42)
    data = {
        'origin': np.random.choice(['CityA', 'CityB', 'CityC'], num_samples),
        'destination': np.random.choice(['CityA', 'CityB', 'CityC'], num_samples),
        'distance': np.random.uniform(10, 100, num_samples),
        'traffic_level': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'travel_time': np.random.uniform(20, 120, num_samples)  # Temps en minuts
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """ Codifica categories i normalitza les distàncies. """
    label_encoder = LabelEncoder()
    df['traffic_level'] = label_encoder.fit_transform(df['traffic_level'])
    scaler = StandardScaler()
    df['distance'] = scaler.fit_transform(df[['distance']])
    return df

def train_model(df):
    """ Entrena un model de regressió per predir el temps de viatge. """
    X = df[['distance', 'traffic_level']]
    y = df['travel_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
    return model

def create_graph(df):
    """ Crea un graf amb nodes i arestes basats en les dades. """
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_edge(row['origin'], row['destination'], weight=row['travel_time'])
    return G

def optimize_routes(G, df, model):
    """ Optimitza el graf actualitzant els pesos amb prediccions del model. """
    for u, v, d in G.edges(data=True):
        row = df[(df['origin'] == u) & (df['destination'] == v)].iloc[0]
        pred_input = np.array([[row['distance'], row['traffic_level']]])
        G[u][v]['weight'] = model.predict(pred_input)[0]
    return G

def find_optimal_route(G, origin, destination):
    """ Troba la ruta més ràpida utilitzant l'algoritme de camí més curt. """
    return nx.shortest_path(G, source=origin, target=destination, weight='weight')

def display_results(route):
    """ Mostra la ruta òptima. """
    print("Ruta òptima:", " -> ".join(route))

# Flux principal del programa
data = generate_fake_data(100)
processed_data = preprocess_data(data)
model = train_model(processed_data)
graph = create_graph(processed_data)
optimized_graph = optimize_routes(graph, processed_data, model)
optimal_route = find_optimal_route(optimized_graph, 'CityA', 'CityB')
display_results(optimal_route)

# Opcional: Visualitza el graf
nx.draw_networkx(optimized_graph, with_labels=True, node_color='lightblue')
plt.show()
