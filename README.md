## **Clasificación de Pollos: Detección de Salud mediante Deep Learning**

### **Descripción**
Este proyecto utiliza redes neuronales convolucionales (CNN) para clasificar pollos como "sanos" o "enfermos" basándose en imágenes y videos. El flujo de trabajo incluye el preprocesamiento de datos, entrenamiento del modelo y generación de métricas para evaluar el desempeño. El objetivo principal es proporcionar una herramienta efectiva para la industria avícola que facilite la detección temprana de enfermedades en pollos.

---

### **Contenido del Proyecto**
1. `data_processing.py`:
   - Procesa videos y extrae frames relevantes.
   - Organiza imágenes etiquetadas en un formato listo para el entrenamiento.
2. `train_model.py`:
   - Entrena un modelo de red neuronal convolucional.
   - Genera una curva de aprendizaje.
   - Incluye métricas de evaluación como precisión, matriz de confusión y reporte de clasificación.
3. `classify_image.py`:
   - Carga un modelo preentrenado y clasifica imágenes individuales como "sano" o "enfermo".

---

### **Estructura del Proyecto**
```
📂 Clasificación de Pollos
├── dataset/
│   ├── raw_videos/         # Carpeta con videos originales
│   ├── images/             # Carpeta con imágenes etiquetadas
│   └── train/              # Carpeta para datos procesados
├── models/
│   └── chicken_health_model.keras  # Modelo entrenado
├── results/
│   ├── learning_curve.png  # Curva de aprendizaje
│   └── evaluation_report.txt  # Reporte de evaluación
├── scripts/
│   ├── data_processing.py
│   ├── train_model.py
│   └── classify_image.py
├── README.md               # Archivo de descripción del proyecto
└── requirements.txt        # Dependencias del proyecto
```

---

### **Requisitos**
- **Python 3.8 o superior**
- Dependencias del proyecto listadas en `requirements.txt`. Instálalas ejecutando:
  ```bash
  pip install -r requirements.txt
  ```

#### **Dependencias principales**
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- OpenCV

---

### **Instrucciones de Uso**

#### **1. Preprocesar Datos**
Ejecuta el script para procesar los datos de entrada (videos e imágenes) y organizarlos en carpetas para el entrenamiento:
```bash
python scripts/data_processing.py
```

#### **2. Entrenar el Modelo**
Entrena la red neuronal utilizando los datos preprocesados:
```bash
python scripts/train_model.py
```
- Esto generará el modelo entrenado (`chicken_health_model.keras`) y la curva de aprendizaje (`learning_curve.png`).

#### **3. Clasificar Imágenes**
Usa el modelo entrenado para clasificar imágenes individuales:
```bash
python scripts/classify_image.py
```
Asegúrate de especificar la ruta del modelo y la imagen en el script antes de ejecutarlo.

---

### **Métricas de Desempeño**
- Precisión: 92%
- F1-Score: 0.91
- Área bajo la curva ROC (AUC-ROC): 0.94

---

### **Resultados**
#### **Curva de Aprendizaje**
La curva de aprendizaje muestra la evolución de la precisión y la pérdida durante el entrenamiento y la validación:

![Curva de aprendizaje](results/learning_curve.png)

#### **Reporte de Evaluación**
El modelo mostró un buen desempeño en el conjunto de validación, con métricas detalladas disponibles en `evaluation_report.txt`.

---

### **Contribución**
Si deseas contribuir al proyecto:
1. Haz un fork del repositorio.
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza un pull request.

---

### **Licencia**
Este proyecto está bajo la licencia [MIT](LICENSE). Puedes utilizarlo libremente con atribución al autor.

---

### **Contacto**
- **Autor**: Marcos Hernandez, Mario Obed Guitiz
- **Correo**: [tuemail@example.com](mailto:tuemail@example.com)
- **GitHub**: [github.com/tuusuario](https://github.com/tuusuario)

