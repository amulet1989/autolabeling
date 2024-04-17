# Autodestilación con Grounding DINO
La autodestilación utiliza un modelo base en nuestro caso [Grounding DINO](https://arxiv.org/pdf/2303.05499.pdf) y un modelo target. La idea general es omitir el proceso de etiquetado manual, dejándole esta tarea al modelo base, el cual generará las anotaciones que necesita el modelo target para entrenarse. 

![Autodistill](https://i.imgur.com/qqmdE2B.jpg)

El modelo base es un modelo altamente preciso, pero pesado para utilizarse en producción para tiempo real, el modelo target por su parte es ligero y rápido, acto para dispositivos de borde y que funcionen en tiempo real. Algunos posibles modelo target son las versiones de YOLO y DETR. 

## Grounded DINO
El modelo [DINO](https://arxiv.org/pdf/2203.03605.pdf) (DETR with Improved deNoising anchOr boxes) es un detetector extremo a extremo de ultima generación basado en transformer. DINO mejora el rendimiento y la eficiencia de modelos anteriores de tipo DETR mediante el uso de un método contrastivo para el entrenamiento por denoising, un método mixto de selección de consultas para la inicialización de anclajes y un esquema de doble búsqueda para la predicción de boxes.

[Grounding DINO](https://arxiv.org/pdf/2303.05499.pdf) es una extensión del modelo DINO que se utiliza para la **detección de objetos de conjunto abierto** en la visión por computadora. Grounding DINO utiliza una combinación de técnicas de procesamiento de lenguaje natural y visión por computadora para mejorar la precisión de la detección de objetos y la comprensión de expresiones de referencia.

Aplica una arquitectura de doble codificador y decodificador simple, lo que significa que tiene codificadores separados para las entradas de imagen y texto y un único decodificador que genera la salida final.

![Arquitectura](https://i.imgur.com/REICgNm.png)

## Instalación
- 1- Clonar el repositorio https://github.com/amulet1989/autolabeling
- 2- Generar un .venv y activarlo usando

```
python -m venv .venv
.\.venv\Scripts\activate
```
- 3- instalar las dependencias
```
pip install -r requirements.txt
```
- 4- asegurarse de tener una carpeta "videos" con subcarpetas con los videos de cada producto. Cada carpeta de producto contendrá tres videos de dos minutos más un archivo *ontology.json* con la ontología necesaria definiendo que queremos que el modelo detecte. La onología contiene el diccionario ontológico que tiene el formato {caption0: class0, captio1:class1...} donde *caption* es la indicación enviada al modelo base, y *class* es la etiqueta que se guardará para ese caption en las anotaciones generadas

Ejemplo de archivo ontology.json:
```
{
  "product held by": "7790040133488"
}
```
- 5- Correr el pipeline dentro del script *autolabeling.py* con los argumetos que estimemos convenientes. Para una prueba inicial vamos a elegir un frame_rate alto para obtener pocas imagenes por video:

```
python autolabeling.py --frame_rate 350

```
Finalmente se generará la carpeta *"dataset"* con subcarpetas (un dataset para cada producto) y la carpeta Merged_Dataset con un dataset de todos los objetos individuales listo para entrenar un modelo YOLO o DETR. 

## Uso de autolabeling.py
La función de autolabeling contiene numerosos parámetros para modificar su funcionameinto:
```
        "--video",
        default=config.VIDEO_DIR_PATH,
        type=str,
        help="Ruta a la carpeta con los videos",
 ``` 
 ```  
        "--image_dir",
        default=config.IMAGE_DIR_PATH,
        type=str,
        help="Ruta al directorio de imágenes",
```
```
        "--frame_rate", default=1, 
        type=int, 
       help="cada cuantos frames toma uno y lo salva como imagen"
 ```
 ```  
        "--ontology",
        default="ontology.json",
        type=str,
      help="Nombre del JSON donde está la ontología, dentro de cada carpeta de video a convertir debe estar el archivo json con   las indicaciones para el modelo",
```
 ```   
        "--output_dataset",
        default=config.DATASET_DIR_PATH,
        type=str,
        help="Carpeta de salida para el dataset de imágenes anotadas",
```
```
    "--extension", 
    default=".jpg", 
    type=str, help="Extensión de los frames que se guardad con imágenes"
 ```
 ```   
        "--box_threshold", 
        default=0.35, 
        type=float, 
        help="Umbral de BBox esto es un parámetro de Grounded DINO que marca el valor de confianza para aceptar que es un BBox correctamente seleccionado"
```
```
        "--text_threshold", 
        default=0.25, 
        type=float, 
        help="Umbral de texto esto es un parámetro de Grounded DINO que marca el valor de confianza para aceptar que es un objeto correctamente relacionado con el prompt"
``` 
```       
        "--max_size", 
        default=0.5, 
        type=float, 
        help="Tamaño maximo de BBox, útil para eliminar BBox grandes, si consideramos que no es lógico tener objetos mayores que umbral dado, su valor está normalizado (0-1) con el tamaño de la imagen"
```  
        "--min_size", 
        default=0.05, 
        type=float, 
        help="Tamaño minimo de BBox, si consideramos que no es lógico tener objetos menores que umbral dado, su valor está normalizado (0-1) con el tamaño de la imagen "
    
```  
        "--iou_threshold",
        default=0.4,
        type=float,
        help="IoU máxima permitida de dos BBox, permite eliminar BBox superpuestos si se desea",
 ```
 ```   
    
        "--not_remove_large",
        default=True,
        action="store_false",
        help="Colocar si no deseamos eliminar BBox demasiado grandes",
 ```
 ```
        "--not_remove_small",
        default=True,
        action="store_false",
        help="Colocar si no deseamos eliminar BBox demasiado pequeños",
```
 ```   
        "--not_remove_overlapping",
        default=True,
        action="store_false",
        help="Colocar si no queremos eliminar BBox que se superpongan",
```
```
        "--not_remove_empty",
        default=True,
        action="store_false",
        help="colocar si no queremos eliminar imágenes sin BBox detectados",
```
```
        "--not_remove_multiple",
        default=True,
        action="store_false",
        help="Colocar si no queremos eliminar imágenes con más de un objeto detectado" 
```
```     
 "--not_augment",
        default=True,
        action="store_false",
        help="Si se desea no aumentar el dataset",
```
```
        "--augmented_for",
        default=4,
        type=int,
        help="Proporción en la que se aumentará el dataset",
```
```
        "--use_yolo",
        default=False,
        action="store_true",
        help="Si se desea utilizar un modelo de YOLO para etiquetar, se debe modificar en la función "mypipeline" al inicio de "autolabeling.py", el path en donde se encuentra el modelo yolo en el formato .pt",
        Nota: cuando se utiliza Yolo la ontología no cumple ningún objetivo, el dataset se generará con los nombres de clases que contenga el modelo 
```
```
        "--num_datasets", 
        default=4, 
        type=int, 
      help="Numero de datasets, útil si se conoce que se generarán muchas imágenes (más de mil) por cada carpeta de videos, si se usa     DINO utilizar añadir una partición cada 1000 imágenes, dependiendo del tamaño de estas, así no se satura la memoria"
```
```
        "--height", 
        default=None, 
        type=int, 
        help="Altura para resize de las imagines, solo poner si se quieren generar imágenes de tamaño diferente al de los frames de video"

        "--width",
         default=None, 
         type=int, 
         help="Ancho para resize de las imagines, solo poner si se quieren generar imágenes de tamaño diferente al de los frames de video "
```

Este sería un ejemplo de uso asumiendo que tenemos en el directorio raiz un carpeta "./videos" con subcarpetas con los videos a anotar y el archivo "ontology.json" con las indicaciones. 
```
python autolabeling.py --frame_rate 50 --not_remove_overlapping --not_remove_multiple --not_augment --num_datasets 1 --use_yolo

```
En este caso se tomará una imagen cada 50 frames, no se removerán los BBox superpuesto que determine el modelo, ni aquellas imágenes con múltiples objetos detectados, no se realizará data augmentation, no se dividirán cada subcarpeta de videos en multiples datasets y se utilizará un modelo Yolo en vez del modelo grounded DINO. 