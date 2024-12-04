from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = FastAPI()

# Ruta del modelo preentrenado
MODEL_PATH = "dataset/model/chicken_health_model_smote.h5"

# Cargar el modelo preentrenado
model = tf.keras.models.load_model(MODEL_PATH)

def classify_image(image_path, img_size=(128, 128)):
    """
    Clasifica una imagen utilizando un modelo preentrenado.
    
    Args:
        image_path (str): Ruta de la imagen a clasificar.
        img_size (tuple): Tamaño de las imágenes (ancho, alto).
    
    Returns:
        str: Resultado de la clasificación.
    """
    # Preprocesar la imagen
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar predicción
    prediction = model.predict(img_array)
    return "Enfermo" if prediction[0][0] < 0.5 else "Sano"

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    """
    Recibe un archivo de imagen, lo clasifica y retorna el resultado.
    
    Args:
        file (UploadFile): Archivo de imagen subido por el cliente.
    
    Returns:
        dict: Resultado de la clasificación con la clase y confianza.
    """
    # Guardar la imagen subida en un archivo temporal
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Clasificar la imagen
        result = classify_image(temp_file_path)
        return {"image": file.filename, "classification": result}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Bloque de escucha
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)