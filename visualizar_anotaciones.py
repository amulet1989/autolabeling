import cv2
import os
import argparse
import tkinter as tk
from tkinter import filedialog


def seleccionar_directorio():
    root = tk.Tk()
    root.withdraw()
    directorio_seleccionado = filedialog.askdirectory(title="Seleccionar directorio")
    return directorio_seleccionado


def mostrar_imagenes_y_anotaciones(directorio_imagenes, directorio_labels):
    # Obtener la lista de archivos de imágenes en el directorio
    archivos_imagenes = sorted(os.listdir(directorio_imagenes))

    # Configurar el sistema de logs
    print("Presiona 'q' para salir.")
    for archivo_imagen in archivos_imagenes:
        if archivo_imagen.endswith(".jpg"):
            ruta_imagen = os.path.join(directorio_imagenes, archivo_imagen)
            ruta_label = os.path.join(
                directorio_labels, archivo_imagen.replace(".jpg", ".txt")
            )

            # Leer la imagen
            imagen = cv2.imread(ruta_imagen)
            altura, ancho, _ = imagen.shape

            # Leer las anotaciones
            with open(ruta_label, "r") as file:
                lineas = file.readlines()
                for linea in lineas:
                    valores = [float(val) for val in linea.strip().split()]
                    clase = int(valores[0])
                    x, y, w, h = valores[1], valores[2], valores[3], valores[4]

                    # Convertir las coordenadas normalizadas a píxeles
                    x_pixel = int((x - w / 2) * ancho)
                    y_pixel = int((y - h / 2) * altura)
                    w_pixel = int(w * ancho)
                    h_pixel = int(h * altura)

                    # Dibujar el rectángulo en la imagen
                    cv2.rectangle(
                        imagen,
                        (x_pixel, y_pixel),
                        (x_pixel + w_pixel, y_pixel + h_pixel),
                        (0, 255, 0),
                        2,
                    )

            # Mostrar la imagen
            escala = 1.0
            imagen = cv2.resize(imagen, (int(ancho * escala), int(altura * escala)))
            cv2.imshow("Imagen", imagen)

            # Esperar la tecla 'q' para salir o cualquier otra tecla para continuar
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Permitir al usuario seleccionar el directorio
    directorio_seleccionado = seleccionar_directorio()
    if not directorio_seleccionado:
        print("No se ha seleccionado un directorio. Saliendo.")
        exit()

    # Obtener los paths de las carpetas "images" y "labels"
    directorio_imagenes = os.path.join(directorio_seleccionado, "images")
    directorio_labels = os.path.join(directorio_seleccionado, "labels")

    # Verificar la existencia de las carpetas "images" y "labels"
    if not os.path.exists(directorio_imagenes) or not os.path.exists(directorio_labels):
        print(
            "No se encontraron las carpetas 'images' y 'labels'. Asegúrate de que la estructura sea correcta."
        )
        exit()

    # Ejecutar la función con los directorios proporcionados
    mostrar_imagenes_y_anotaciones(directorio_imagenes, directorio_labels)
