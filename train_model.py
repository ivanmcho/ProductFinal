import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def plot_learning_curve(history, output_path="learning_curve.png"):
    """
    Genera y guarda la curva de aprendizaje basada en el historial de entrenamiento.
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

def train_model_with_smote(train_dir, val_dir=None, img_size=(128, 128), batch_size=32, epochs=10):
    # Generadores de datos
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # Extraer imágenes y etiquetas del generador
    X_train, y_train = [], []
    for i in range(len(train_data)):
        x_batch, y_batch = train_data[i]
        X_train.extend(x_batch)
        y_train.extend(y_batch)
        if len(X_train) >= train_data.samples:
            break

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Aplicar SMOTE para balancear las clases
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten para SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)
    X_resampled = X_resampled.reshape(-1, img_size[0], img_size[1], 3)  # Restaurar dimensiones originales

    # Dividir datos balanceados en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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
    checkpoint = ModelCheckpoint("dataset/model/best_model.keras", save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(patience=5, monitor='val_loss', mode='min')

    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop]
    )

    return model, history, (X_val, y_val)

def evaluate_model(model, val_data):
    """
    Evalúa el modelo en el conjunto de validación y muestra métricas adicionales.
    """
    X_val, y_val = val_data

    # Predicciones
    y_pred = (model.predict(X_val) > 0.5).astype("int32").flatten()

    # Reporte de clasificación
    print("\nMétricas de clasificación:")
    print(classification_report(y_val, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_val, y_pred)
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

        model, history, val_data = train_model_with_smote(train_dir, val_dir, img_size, batch_size, epochs)

        # Guardar el modelo
        model.save("dataset/model/chicken_health_model_smote.h5")
        print("Modelo guardado como 'chicken_health_model_smote.h5'.")

        # Generar y guardar la curva de aprendizaje
        plot_learning_curve(history, "learning_curve_smote.png")

        # Evaluar el modelo
        print("\nEvaluación del modelo en el conjunto de validación:")
        evaluate_model(model, val_data)

    except Exception as e:
        print(f"Error durante el entrenamiento o evaluación: {e}")