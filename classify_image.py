import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def classify_image(model_path, image_path, img_size=(128, 128)):
    """
    Clasifica una imagen utilizando un modelo preentrenado.
    
    Args:
        model_path (str): Ruta del modelo guardado.
        image_path (str): Ruta de la imagen a clasificar.
        img_size (tuple): Tamaño de las imágenes (ancho, alto).
    
    Returns:
        str: Resultado de la clasificación.
    """
    # Cargar el modelo
    model = tf.keras.models.load_model(model_path)

    # Preprocesar la imagen
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar predicción
    prediction = model.predict(img_array)
    return "Enfermo" if prediction[0][0] < 0.5 else "Sano"

# Main
if __name__ == "__main__":
    model_path = "dataset/model/chicken_health_model_smote.h5"
    values_folder = "values"
    
    # Obtener la lista de archivos en la carpeta "values"
    image_files = [f for f in os.listdir(values_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No se encontraron imágenes en la carpeta 'values'.")
    else:
        for image_file in image_files:
            image_path = os.path.join(values_folder, image_file)
            print(f'Imagen de entrada: {image_path}')
            try:
                result = classify_image(model_path, image_path)
                print(f"El pollo en la imagen '{image_file}' está: {result}")
            except Exception as e:
                print(f"Error procesando la imagen '{image_file}': {e}")