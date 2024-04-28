import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

def leer_datos_archivo(archivo):
    """
    Lee datos desde un archivo de texto y los convierte a una lista de floats.
    
    Parámetros:
        archivo (str): Ruta al archivo de texto.
        
    Retorna:
        list: Lista de valores flotantes.
    """
    with open(archivo, 'r') as file:
        datos = file.readlines()
        datos = [float(line.strip()) for line in datos if line.strip()]
    return datos

def generar_datos_sinteticos(velocidades, tiempos, consumos, num_muestras=100, desviacion=0.1):
    velocidades = np.array(velocidades)
    tiempos = np.array(tiempos)
    consumos = np.array(consumos)
    datos_sinteticos = np.zeros((num_muestras, 3))
    
    for i in range(num_muestras):
        idx = np.random.randint(len(velocidades))
        datos_sinteticos[i, 0] = velocidades[idx] * (1 + np.random.normal(0, desviacion))
        datos_sinteticos[i, 1] = tiempos[idx] * (1 + np.random.normal(0, desviacion))
        datos_sinteticos[i, 2] = consumos[idx] * (1 + np.random.normal(0, desviacion))
        
    return datos_sinteticos

def guardar_datos_sinteticos(datos, archivo="dades.txt"):
    """
    Guarda los datos sintéticos en un archivo de texto.
    
    Parámetros:
        datos (np.array): Matriz de datos sintéticos.
        archivo (str): Nombre del archivo para guardar los datos.
    """
    with open(archivo, 'w') as file:
        for fila in datos:
            line = ' '.join(map(str, fila))
            file.write(line + '\n')

def bootstrap_regression(X, y, n_bootstraps=1000):
    coefs = []
    for _ in range(n_bootstraps):
        X_sample, y_sample = resample(X, y)
        model = LinearRegression().fit(X_sample, y_sample)
        coefs.append(model.coef_)
        
    return np.array(coefs)

# Pedir al usuario que especifique el número de muestras a generar
num_muestras = int(input("Ingrese el número de muestras sintéticas a generar: "))

# Leer los datos de los archivos
velocidades = leer_datos_archivo("velocitat.txt")
tiempos = leer_datos_archivo("temps.txt")
consumos = leer_datos_archivo("consum.txt")

# Generar datos sintéticos
datos_sinteticos = generar_datos_sinteticos(velocidades, tiempos, consumos, num_muestras=num_muestras)

# Guardar los datos sintéticos generados en un archivo llamado 'dades.txt'
guardar_datos_sinteticos(datos_sinteticos)

X = datos_sinteticos[:, :2]  # Velocidades y tiempos
y = datos_sinteticos[:, 2]   # Consumos

# Realizar Bootstrapping en el modelo de regresión
coeficientes_bootstrap = bootstrap_regression(X, y, n_bootstraps=1000)

# Calcular la media y el intervalo de confianza de los coeficientes
coef_mean = np.mean(coeficientes_bootstrap, axis=0)
coef_ci_low, coef_ci_high = np.percentile(coeficientes_bootstrap, [2.5, 97.5], axis=0)

print("Coeficientes medios:", coef_mean)
print("Intervalos de confianza al 95% para los coeficientes:", list(zip(coef_ci_low, coef_ci_high)))
