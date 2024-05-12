import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import heapq
import scipy as sp
from graphviz import Graph
from pyvis.network import Network
from collections import defaultdict

# PRE: El fitxer "filename" existeix i conté línies amb el format "puntA, puntB, distancia, temps, seguretat".
#      La línia "END OF INPUT" marca la fi del fitxer.
# POST: S'ha creat un diccionari "adjacency_list" que representa la llista d'adjacència.
#       Cada clau correspon a un punt del graf, i els seus valors són tuples amb els punts adjacents i els seus pesos associats.
# Cost espai-temporal: O(E), on E és el nombre d'arestes (relacions d'adjacència) al graf.
def read_file(filename: str = "input1.txt") -> dict:
    """
    Llegeix un fitxer de text i crea una llista d'adjacència per a un graf no dirigit.

    :param filename: El nom del fitxer a llegir (per defecte: "input1.txt").
    :return: Un diccionari que representa la llista d'adjacència.
    """
    adjacency_list = defaultdict(list) # Diccionari amb llistes com a valors per a cada clau
    distance_list = defaultdict(list)
    time_list = defaultdict(list)
    safety_list = defaultdict(list)
    with open(filename, 'r') as f:
        # Invariant: El bucle "for line in f" recorre totes les línies del fitxer.
        for line in f:
            line = line.strip() # Elimina espais en blanc al principi i al final de la línia
            if line != 'END OF INPUT':
                # Invariant: El bucle "for pointA, pointB, weight in line.split()" divideix cada línia en tres parts.
                pointA, pointB, distance, time, safety = line.split(", ") # Divideix la línia en tres parts
                distance = float(distance) # Converteix el pes a un enter
                time = float(time)
                safety = float(safety)
                
                # Calcula el peso heurístico combinando distancia, tiempo y seguridad (suma ponderada)
                h_weight = heuristic_graph(distance, time, safety, 0.3, 0.6, 0.1)
               
                # Afegeix una tupla (pointB, weight) a la llista d'adjacència de pointA i viceversa
                adjacency_list[pointA].append((pointB, float(f"{h_weight:.4f}")))
                adjacency_list[pointB].append((pointA, float(f"{h_weight:.4f}")))
                
                # Afegeix una tupla (pointB, distance) a la distance_list de pointA i viceversa
                distance_list[pointA].append((pointB, distance))
                distance_list[pointB].append((pointA, distance))
                
                # Afegeix una tupla (pointB, time) a la time_list de pointA i viceversa
                time_list[pointA].append((pointB, time))
                time_list[pointB].append((pointA, time))
                
                # Afegeix una tupla (pointB, safety) a la safety_list de pointA i viceversa
                safety_list[pointA].append((pointB, safety))
                safety_list[pointB].append((pointA, safety))
                
    return adjacency_list, distance_list, time_list, safety_list

def uniform_cost_search(adjacency_list: dict, origin: str, destination: str) -> list:
    """
    Realiza una búsqueda de costo uniforme en un grafo representado por una lista de adyacencia.

    Args:
        adjacency_list (dict): Lista de adyacencia del grafo.
        origin (str): Nodo de inicio de la búsqueda.
        destination (str): Nodo de destino de la búsqueda.

    Returns:
        list: Lista que representa el camino óptimo desde el nodo de origen hasta el nodo de destino.
    """

    # Inicializar una cola de prioridad con el nodo inicial y un costo de 0
    priority_queue = [(0, origin, [origin])]
    # Inicializar un conjunto de nodos visitados
    visited = set()

    while priority_queue:
        # Eliminar el nodo con el menor costo de la cola de prioridad
        cost, node, path = heapq.heappop(priority_queue)
	
	    # Si se alcanza el nodo destino, devolver el camino desde el nodo origen hasta el nodo destino
        if node == destination:
            if cost > 1.0:
                return [], 1
            else:
                return path, cost

        if node not in visited:
            # Marcar el nodo como visitado
            visited.add(node)
            # Agregar los vecinos a la cola de prioridad con sus costos correspondientes
            for neighbor, weight in adjacency_list[node]:
                if neighbor not in visited:
                    # Calcular el costo del vecino como la suma del costo del nodo actual y 
                    # el peso de la arista entre el nodo actual y el vecino
                    neighbor_cost = cost + weight
                    # Agregar el vecino a la cola de prioridad con su costo y camino correspondientes
                    heapq.heappush(priority_queue, (neighbor_cost, neighbor, path + [neighbor]))

    # Si no se alcanza el nodo destino, devolver un camino vacío
    return [], 0


# PRE: Las variables 'distance', 'time' y 'safety' representan la distancia,
#      el tiempo y el nivel de seguridad de una ruta respectivamente.
#      Las variables 'weight_distance', 'weight_time' y 'weight_safety'
#      representan los pesos asignados a la distancia, el tiempo y la seguridad
#      en el cálculo heurístico.
# POST: Retorna el valor heurístico calculado en base a los valores proporcionados.
# COSTO: O(1), complejidad temporal constante.
def heuristic_graph(distance, time, safety, w_distance, w_time, w_safety):
    """
    Calcula el valor heurístico para una ruta dada.

    Args:
        distance (float): La distancia de la ruta.
        time (float): El tiempo requerido para la ruta.
        safety (float): El nivel de seguridad de la ruta.
        weight_distance (float): El peso asignado a la distancia en el cálculo heurístico.
        weight_time (float): El peso asignado al tiempo en el cálculo heurístico.
        weight_safety (float): El peso asignado a la seguridad en el cálculo heurístico.

    Returns:
        float: El valor heurístico para la ruta dada.
    """
    MAX_DISTANCE = 250  # El valor máximo posible de la distancia
    MAX_TIME = 3  # El valor máximo posible del tiempo
    MAX_SAFETY = 10  # El valor máximo posible de la seguridad

    # Normalizar los valores
    distance_normalized = distance / MAX_DISTANCE
    time_normalized = time / MAX_TIME
    safety_normalized = safety / MAX_SAFETY
    
    # Calcular el valor heurístico
    heuristic_value = (w_distance * distance_normalized + w_time * time_normalized + w_safety * safety_normalized)

    return heuristic_value


# PRE: La variable 'adjacency_list' es un diccionario que representa la lista de adyacencia del grafo.
#      La variable 'path' es una lista que contiene la ruta óptima en el grafo, si se proporciona.
# POST: Se ha generado y mostrado el grafo en un archivo HTML.
# Costo espacial-temporal: O(V + E), donde V es el número de vértices y E es el número de aristas en el grafo.
def plot_graph(adjacency_list: dict, path: list = []):
    """
    Genera y muestra un grafo visualmente utilizando la biblioteca pyvis.

    Args:
        adjacency_list (dict): Diccionario que representa la lista de adyacencia del grafo.
        path (list, optional): Ruta óptima a resaltar en el grafo. Por defecto, una lista vacía.

    Returns:
        None
    """
    # Crear un objeto Network para visualizar el grafo
    net = Network(notebook=True, height="750px", width="100%", cdn_resources='in_line') 
    # Se configura para ser mostrado en un entorno de notebook, con una altura de 750px y un ancho del 100% 
    # de la ventana del navegador. Además se especifica que los recursos necesarios para visualizar el grafo 
    # se incrustarán directamente en el HTML generado.   

    # Agregar nodos al grafo
    for node in adjacency_list.keys():
        net.add_node(node, label=node, title=node) 
        # Invariante: este bucle itera sobre cada nodo en el diccionario “adjacency_list” y agrega cada 
        # nodo al grafo “net”. El “label” y el “title” de cada nodo se establecen en el nombre del nodo.

    # Agregar aristas al grafo
    for node, connections in adjacency_list.items():
        for connection, weight in connections:
            net.add_edge(node, connection, label=str(weight), title=str(weight)) 
            # Invariante: este bucle anidado itera sobre cada nodo en el diccionario “adjacency_list” 
            # y sus conexiones. Para cada conexión, agrega una arista al grafo “net”. El “label” y el 
            # “title” de cada arista se establecen en el peso de la conexión.

    # Resaltar los nodos y aristas en la ruta óptima, si se proporciona
    if path:
        for node in path:
            net.get_node(node)["color"] = "green" 
            # Invariante: este bucle itera sobre cada nodo en la lista “path”, estableciendo el color 
            # de cada nodo en verde.
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            for edge in net.edges:
                if edge["from"] == from_node and edge["to"] == to_node or edge["from"] == to_node and edge["to"] == from_node:
                    edge["color"] = "red" 
                    # Invariante: este bucle itera sobre cada par de nodos consecutivos en “path”, 
                    # encontrando la arista correspondiente en el grafo y estableciendo su color en rojo
                    break

    # Generar y mostrar el grafo en un archivo HTML
    net.show("graph.html")


# PRE: La variable 'adjacency_list' es un diccionario que representa la lista de adyacencia del grafo.
#      La variable 'path' es una lista que contiene la ruta óptima en el grafo, si se proporciona.
# POST: Se ha generado y mostrado el grafo en un archivo HTML.
# Costo espacial-temporal: O(V + E), donde V es el número de vértices y E es el número de aristas en el grafo.
def plot_optimal_route(adjacency_list: dict, name: str, path: list = []):
    """
    Genera y muestra un grafo visualmente utilizando la biblioteca pyvis,
    resaltando solo los nodos y aristas que forman parte de la ruta óptima.

    Args:
        adjacency_list (dict): Diccionario que representa la lista de adyacencia del grafo.
        path (list): Lista que representa la ruta óptima en el grafo.

    Returns:
        None
    """
    # Crear un objeto Network para visualizar el grafo
    net = Network(notebook=True, height="750px", width="100%", cdn_resources='in_line')
    
    # Agregar nodos y aristas que forman parte de la ruta óptima
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
    
        # Agregar el nodo de origen si aún no ha sido agregado
        if from_node not in net.get_nodes():
            net.add_node(from_node, label="origen: "+from_node, title=from_node, color="green")

        # Agregar el nodo de destino si aún no ha sido agregado
        if to_node not in net.get_nodes():
            net.add_node(to_node, label=to_node, title=to_node, color="green")
 
        # Agregar la arista entre el nodo de origen y el nodo de destino
        for connection, weight in adjacency_list[from_node]:
            if connection == to_node:
                net.add_edge(from_node, connection, label=str(weight), title=str(weight), color="red")

    # Mostrar el grafo en un archivo HTML
    net.show(name)


def user_decide_next_move(adjacency_list: dict, origin: str, destination: str, optimal_path):
    """
    Interactúa con el usuario para decidir el próximo destino en un recorrido, considerando o recalculando
    la ruta óptima según las elecciones del usuario.

    Parámetros:
    - adjacency_list: Diccionario que representa el grafo de ciudades conectadas.
    - origin: La ciudad de origen o punto de partida actual.
    - destination: La ciudad destino o punto final deseado.
    - optimal_path: Lista que representa la ruta óptima calculada desde el origen hasta el destino.

    La función continúa solicitando al usuario que elija su próximo movimiento hasta que se alcance el destino.
    Si el usuario elige el siguiente movimiento correcto según la ruta óptima, se avanza sin recalcular la ruta.
    De lo contrario, si elige un camino diferente, se recalcula y muestra una nueva ruta óptima desde la nueva posición actual hasta el destino.
    """
    current_origin = origin  # Ciudad actual de partida
    current_path_index = 0  # Índice de la ciudad actual en la ruta óptima

    while current_origin != destination: # Verificar si hay movimientos posibles desde la ubicación actual
        if current_origin in adjacency_list:
            possible_moves = adjacency_list[current_origin]
            print(f"\nTu siguiente movimiento debería ser: {optimal_path[current_path_index + 1]}")
            print(f"\nDesde {current_origin}, los movimientos posibles son:")
            for i, move in enumerate(possible_moves, start=1):
                next_destination, distance = move
                print(f"{i}. A {next_destination} con una distancia de {distance} km")

        	# Solicitar al usuario que elija su próximo destino
            choice = int(input("Elige tu próximo destino (número): ")) - 1
            if 0 <= choice < len(possible_moves):
                next_destination, _ = possible_moves[choice]
                print(f"\nHas elegido ir a {next_destination}.")

            	# Comprobar si la elección del usuario coincide con la ruta óptima
                if next_destination == optimal_path[current_path_index + 1]:
                    print("Has elegido el movimiento correcto según la ruta óptima.")
                    current_origin = next_destination
                    current_path_index += 1
                    if current_origin == destination:
                        print("Has llegado a tu destino final.")
                        return
                else:
                    # Recalcular y mostrar la nueva ruta óptima si el usuario elige un camino diferente
                    new_path = uniform_cost_search(adjacency_list, next_destination, destination)
                    if new_path:
                        print("Ruta recalculada hacia el destino final:", " -> ".join(new_path))
                        plot_graph(adjacency_list, path=new_path)
                        optimal_path = new_path
                        current_path_index = 0
                    else:
                        print("No se pudo encontrar una ruta desde tu ubicación actual hasta el destino final.")
                        return
                current_origin = next_destination
            else:
                print("Opción no válida, intenta de nuevo.")
        else:
            print(f"No se encontraron movimientos posibles desde {current_origin}.")
            return


def main():
    # Esborra la pantalla per a una millor visualització
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Sol·licita a l'usuari la ciutat d'origen
    origin = str(input("Escriu el punt d'origen: "))  
    
    # Sol·licita a l'usuari la ciutat de destí
    destination = str(input("Escriu el punt de desti: ")) 

    # Defineix el nom del fitxer que conté les dades del graf
    filename = "inputs/vilanova.txt"
    
    # Llegeix les dades del fitxer i crea la llista d'adjacència del graf
    adjacency_list, distance_list, time_list, safety_list = read_file(filename)
    
    # Indica que es calcula la ruta òptima
    print("Calculant la ruta òptima...")

    
    # Calcula la ruta òptima des de l'origen fins al destí
    optimal_path, costo_total = uniform_cost_search(adjacency_list, origin, destination)
    
    #print("DL: ", optimal_path)
    #distance = next((dist for city, dist in distance_list[optimal_path[0]] if city == optimal_path[1]), None)
    #print(f"The distance between Avinguda del Mar and Carrer del Gran Passeig Marítim is {distance} km")
    
    total_distance = 0
    total_time = 0
    total_safety = 0
    
    # busca en les llistes de distace, time i safety les parelles de carrers i en va sumant el valor numeric que tenen
    # km si es distancia, min si es temps i el valor de seguretat del carrer
    for i in range(len(optimal_path) - 1):
        start = optimal_path[i]
        end = optimal_path[i + 1]
        # Find distance between start and end cities if it exists
        distance = next((dist for city, dist in distance_list[start] if city == end), None)
        time = next((time for city, time in time_list[start] if city == end), None)
        safety = next((safety for city, safety in safety_list[start] if city == end), None)
        if distance and time and safety is not None:
            total_distance += distance
            total_time += time
            total_safety += safety
        else:
            print(f"No direct connection found between {start} and {end}")
            
    # Round the total distance to two decimal places
    total_distance_rounded = round(total_distance, 2)
	
    if optimal_path:
        # Imprimeix la ruta òptima teòrica des de l'origen fins al destí
        print(f"La ruta òptima desde {origin} fins {destination} és:")
        for i, nodo in enumerate(optimal_path, start=1):
            print(f"   {i}. {nodo}")
        print("\nCost total de la ruta òptima: ", costo_total)
        
        print(f"La distancia total de la ruta optima es {total_distance_rounded} km")
        print(f"El temps total de la ruta optima es {total_time} min")
        print(f"L'indicador de seguretat de la ruta optima es {total_safety}")
        
        # Visualitza el graf només de la ruta òptima
        plot_optimal_route(adjacency_list, "optimal_route.html", path=optimal_path)
        # Visualitza el graf sencer amb la ruta òptima
        plot_graph(adjacency_list, path=optimal_path)
        
        # Visualitza el graf només de la ruta òptima amb la distancia
        plot_optimal_route(distance_list, "distance_route.html", path=optimal_path)
        # Visualitza el graf només de la ruta òptima amb el temps
        plot_optimal_route(time_list, "time_route.html", path=optimal_path)
        # Visualitza el graf només de la ruta òptima amb el safety
        plot_optimal_route(safety_list, "safety_route.html", path=optimal_path)
        
    else:
        if costo_total == 0:
            # Indica que no s'ha trobat cap ruta òptima
            print("No s'ha trobat una ruta òptima.")
        else:
	    # Indica que el cost heuristic es major que 1
            print("La ruta òptima supera l'autonomia del vehicle elèctric-autonom.")
        return
    
    # Permet a l'usuari decidir els següents moviments
    #user_decide_next_move(adjacency_list, origin, destination, optimal_path)  
if __name__ == '__main__':
    # Inicia l'execució del programa principal si s'executa com a script
    main()  
