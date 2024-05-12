from mapbox import Directions

# Codi d'accés al token de MapBox que descarrega les dades que necessitem de l'API de MapBox
token = 'sk.eyJ1IjoibWFyY21ydXBjIiwiYSI6ImNsdmlhbGdweTA3enoyanF2cXNvcnFsMHMifQ.voJfI2ybCE-hN12f08M9zw'

# Guardem a la variable directions el dataset que descarreguem de MapBox
directions = Directions(access_token=token)

# Coordenades com a exemple de dos punts diferents del mapa (longitud, latitud)
point_a = (-74.0059, 40.7128)  # Nueva York
point_b = (-118.2437, 34.0522)  # Los Ángeles

try:
    # Amb aquesta funció obtenim la ruta que volem realitzar i les dades de temps i distancia d'aquesta ruta
    response = directions.directions([point_a, point_b], 'mapbox/driving', steps=False)

    # Verifiquem si l'intent d'agafar les dades del dataset de MapBox ha sigut correcta o no
    if response.status_code == 200:
        # Guardem en variables independents la distancia i el temps obtinguts anteriorment
        distance = response.json()['routes'][0]['distance']  # en metres
        duration = response.json()['routes'][0]['duration']  # en segons

        # Expressa la distancia abtinguda en kilometres (la dada be de MapBox en metres) 
	 # Expressa el temps obtingut en minuts (la dada be de MapBox en segons)
        distance_km = distance / 1000
        duration_min = duration / 60

        print(f"Distancia: {distance_km:.2f} km")
        print(f"Temps: {duration_min:.2f} minuts")
    else:
        print("Error en la solicitud:", response.status_code)
except Exception as e:
    print("Ocurrió un error:", e)