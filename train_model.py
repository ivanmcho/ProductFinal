import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix

def plot_learning_curve(history, output_path="learning_curve.png"):
    """
    Genera y guarda la curva de aprendizaje basada en el historial de entrenamiento.
    
    Args:
        history: Historial de entrenamiento del modelo (objeto devuelto por model.fit()).
        output_path (str): Ruta donde se guardará la imagen de la curva.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Precisión Entrenamiento")
    plt.plot(epochs, val_acc, label="Precisión Validación")
    plt.title("Precisión del Modelo")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Pérdida Entrenamiento")
    plt.plot(epochs, val_loss, label="Pérdida Validación")
    plt.title("Pérdida del Modelo")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Curva de aprendizaje guardada en: {output_path}")

def train_model(train_dir, val_dir=None, img_size=(128, 128), batch_size=32, epochs=10):
    # Generadores de datos
    datagen_train = ImageDataGenerator(rescale=1.0/255, validation_split=0.2 if not val_dir else 0.0)
    datagen_val = ImageDataGenerator(rescale=1.0/255)

    train_data = datagen_train.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training' if not val_dir else None
    )
    val_data = datagen_train.flow_from_directory(
        train_dir if not val_dir else val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation' if not val_dir else None
    )

    # Construir modelo
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(patience=5, monitor='val_loss', mode='min')

    # Entrenar modelo
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )

    return model, history, val_data
def evaluate_model(model, val_data):
    """
    Evalúa el modelo en el conjunto de validación y muestra métricas adicionales.
    
    Args:
        model: Modelo entrenado.
        val_data: Generador de datos de validación.
    """
    # Evaluación en datos de validación
    y_true = val_data.classes
    y_pred = (model.predict(val_data) > 0.5).astype("int32").flatten()

    # Reporte de clasificación
    print("\nMétricas de clasificación:")
    print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

# Main
if __name__ == "__main__":
    try:
        train_dir = "dataset/train"
        val_dir = None
        img_size = (128, 128)
        batch_size = 32
        epochs = 10

        model, history, val_data = train_model(train_dir, val_dir, img_size, batch_size, epochs)

        # Guardar el modelo
        model.save("chicken_health_model.h5")
        print("Modelo guardado como 'chicken_health_model.h5'.")

        # Generar y guardar la curva de aprendizaje
        plot_learning_curve(history, "learning_curve.png")

        # Evaluar el modelo
        print("\nEvaluación del modelo en el conjunto de validación:")
        evaluate_model(model, val_data)

    except Exception as e:
        print(f"Error durante el entrenamiento o evaluación: {e}")
