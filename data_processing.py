import cv2
import os
import shutil

def process_videos_and_images(video_dir, image_dir, output_dir, frame_interval=30):
    """
    Procesa videos e imágenes para generar un conjunto de datos listo para el entrenamiento.
    
    Args:
        video_dir (str): Carpeta con los videos organizados por etiquetas.
        image_dir (str): Carpeta con imágenes ya etiquetadas.
        output_dir (str): Carpeta donde se guardarán los datos procesados.
        frame_interval (int): Número de frames entre cada extracción.
    """
    # Asegurar que las carpetas de salida existen
    os.makedirs(output_dir, exist_ok=True)

    # Procesar videos
    for label in os.listdir(video_dir):
        label_video_path = os.path.join(video_dir, label)
        label_output_path = os.path.join(output_dir, label)
        os.makedirs(label_output_path, exist_ok=True)

        for video_file in os.listdir(label_video_path):
            video_path = os.path.join(label_video_path, video_file)
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue  # Ignorar archivos no de video

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(label_output_path, f"{video_file}_frame_{saved_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"Procesado: {video_file}, Frames extraídos: {saved_count}")

    # Copiar imágenes existentes al conjunto de datos
    for label in os.listdir(image_dir):
        label_image_path = os.path.join(image_dir, label)
        label_output_path = os.path.join(output_dir, label)
        os.makedirs(label_output_path, exist_ok=True)

        for image_file in os.listdir(label_image_path):
            if not image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue  # Ignorar archivos no de imagen

            src = os.path.join(label_image_path, image_file)
            dst = os.path.join(label_output_path, image_file)
            shutil.copy(src, dst)

        print(f"Imágenes copiadas de {label}.")

# Main
if __name__ == "__main__":
    video_dir = "dataset/raw_videos"  # Carpeta con videos
    image_dir = "dataset/images"      # Carpeta con imágenes
    output_dir = "dataset/train"      # Carpeta para datos procesados
    frame_interval = 30               # Extraer un frame cada 30 frames

    process_videos_and_images(video_dir, image_dir, output_dir, frame_interval)
    print("Datos listos para el entrenamiento.")