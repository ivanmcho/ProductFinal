import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
    return "Sano" if prediction[0][0] < 0.5 else "Enfermo"

# Main
if __name__ == "__main__":
    model_path = "chicken_health_model.h5"
    image_path = "path_to_your_image.jpg"

    result = classify_image(model_path, image_path)
    print(f"El pollo está: {result}")