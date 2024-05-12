import networkx as nx

def calculate_optimal_route_with_ml(adjacency_list, ml_model):
    """
    Funció per calcular la ruta òptima utilitzant prediccions de ML per actualitzar els pesos.

    Args:
    adjacency_list (dict): Llista d'adjacència del graf.
    ml_model (object): Model de machine learning entrenat.

    Returns:
    tuple: ruta òptima i mètriques associades (distància, temps estimat).
    """
    # Crear un graf a partir de la llista d'adjacència
    G = nx.Graph()
    for origin, edges in adjacency_list.items():
        for destination, distance, time, safety in edges:
            # Preparar les dades per a la predicció del ML
            features = [[distance, time, safety]]  # Aquesta llista ha de coincidir amb les característiques usades per entrenar el ML
            predicted_time = ml_model.predict(features)[0]
            # Afegir aresta amb el temps predit com a pes
            G.add_edge(origin, destination, weight=predicted_time)

    # Suposem que volem calcular la ruta des d'un punt origen fins a un punt destí (definir segons el cas)
    origin = 'A'
    destination = 'B'
    # Calcular la ruta més curta basada en el pes actualitzat
    path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
    path_edges = zip(path, path[1:])
    total_weight = sum(G[u][v]['weight'] for u, v in path_edges)

    return path, total_weight

def display_results(route, metrics):
    """
    Funció per mostrar els resultats de la ruta òptima.

    Args:
    route (list): Llista dels nodes en la ruta òptima.
    metrics (float): Temps total de la ruta (o altres mètriques rellevants).
    """
    print("Ruta Òptima:")
    for index, node in enumerate(route):
        print(f"{index + 1}. {node}")
    print(f"Temps total estimat de la ruta: {metrics:.2f} minuts")

if __name__ == '__main__':
    # Inicia l'execució del programa principal si s'executa com a script
    # Suposant que les dades estan en un fitxer anomenat 'route_data.csv'
    processed_data = collect_and_prepare_data('datos.txt')
    # Suposem que 'adjacency_list' és la teva llista d'adjacència i 'ml_model' és el model entrenat
    route, total_weight = calculate_optimal_route_with_ml(adjacency_list, ml_model)

    # Mostrar la ruta i el temps total
    display_results(route, total_weight)
    main()  
