#predicción-de-accidente-automovilístico Version 1

Este proyecto utiliza los datos de ubicación del vehículo (también conocidos como datos de latitud y longitud o datos de trayectoria) y datos de velocidad y aceleración para caracterizar los hábitos y características de conducción habituales del conductor, y luego predecir la probabilidad de que el vehículo se vea involucrado en una colisión.

### Dificultades del proyecto

1. Las colisiones son muy aleatorias y no está claro si el conductor fue atropellado por otro vehículo o si él mismo chocó contra otro vehículo. También se ve afectado por el clima, las condiciones de la carretera, el estado del conductor en ese momento y factores inciertos, etc.

2. Las funciones básicas del controlador son difíciles de utilizar. Dado que las características básicas del conductor son las registradas cuando se compró el vehículo, es muy probable que el vehículo fuera conducido por otra persona, o que alguien más estuviera conduciendo el vehículo en el momento de la colisión.

3. La cantidad de datos es grande y es difícil procesarlos todos a la vez. La cantidad de datos es relativamente grande y es necesario procesarlos en lotes.

4. Dado que los datos se transmiten al servidor a través de señales 4g, pueden ocurrir interrupciones en la señal, fallas en la carga de datos, etc. En este momento, cómo preprocesar los datos también es un paso muy importante.

5. Los datos de velocidad y aceleración se recopilan solo después de que la aceleración alcanza un cierto umbral, por lo que para los vehículos que viajan a una velocidad constante, no hay registro de velocidad y aceleración. Por el contrario, para algunos vehículos que suelen frenar y acelerar repentinamente, habrá más récords de velocidad y aceleración.

6. A través de la distribución del volumen de datos, encontramos que algunos días, debido a actualizaciones de equipos, solo una cantidad muy pequeña de vehículos tendrá registros ese día. Habrá algunas diferencias en la cantidad de datos.

7. La proporción de muestras positivas y negativas está extremadamente desequilibrada. La tasa de precisión comúnmente utilizada no se puede utilizar para medir el modelo. Es necesario utilizar otros indicadores para entrenar el modelo.

### Ideas de investigación

Primero, haremos una breve introducción basada en los antecedentes y el contenido de los datos del proyecto. Todos los contenidos están en introducción.md.

Luego cuente algunas características antes de la colisión, como la velocidad antes de la colisión, el momento de la colisión, etc. Todo el contenido está en crash_feature.md.

Analizo principalmente datos de posición del vehículo, es decir, datos de trayectoria, y realizo algunas características de ingeniería a través de datos de trayectoria. Las características se extraen de las características temporales y espaciales respectivamente para la predicción. Todo el contenido está en trajectory_feature.md y los códigos de extracción de características están en espacial_feature.py y temporal_feature.

Al hacer referencia a las ideas de investigación del artículo "Usted es cómo conduce: aprendizaje de representación temporal y entre pares para el análisis del comportamiento de conducción", la información de la velocidad y dirección de conducción del vehículo se extrae para formar un estado de tupla de <velocidad, dirección > Hay 9 estados en total, a saber: seguir recto a velocidad constante, girar a la izquierda a velocidad constante, girar a la derecha a velocidad constante, acelerar y seguir recto, acelerar y girar a la izquierda, acelerar y girar a la derecha, desacelerar y seguir. Siga recto, desacelere y gire a la izquierda, y desacelere y gire a la derecha. Mediante el análisis de datos, se construye la matriz de probabilidad de transición y la matriz de tiempo de transición.

Las predicciones se realizan reconstruyendo la función de pérdida mediante el método de incrustar GRU a través del marco del modelo de codificación automática (teniendo en cuenta que el comportamiento de conducción cambia con el tiempo). (La mayor diferencia con el documento es que estamos supervisando el aprendizaje)

Los avances y resultados se actualizan continuamente...