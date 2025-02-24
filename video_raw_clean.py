#!/usr/bin/python3
import cv2
from time import time, sleep
from threading import Lock, Thread
from flask import Flask, Response
import cv2
import numpy as np
from time import time
from threading import Lock
import logging


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Ou logging.CRITICAL para menos logs


# Define a lista de URLs das câmeras
camera_urls = [
    "http://10.1.60.187:5000/video_raw4",  # Indice 0/1
    "http://10.1.60.187:5000/video_raw2",  # Indice 0/1
]

#model = YOLO('yolov8s-pose.pt').to('cpu')
app = Flask(__name__)

# Inicializa os frames e frames anotados globais
global_frames = [np.zeros((320, 480, 3), dtype=np.uint8)] * len(camera_urls)  # Inicializa corretamente
frame_lock = Lock()
annotated_frames = [None] * len(camera_urls)

fps = None

global_fps = []

def imageUpdater(id, video_path, interval):
    """
    Atualiza o frame da câmera, calcula o FPS e desenha-o no frame.
    """
    global global_frames, fps
    
    cap = cv2.VideoCapture(video_path)
    last_time = time()
    start_time = time()
    frame_counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    
    while True:
        current_time = time()
        if current_time - last_time >= interval:
            last_time = current_time
            success, frame = cap.read()
            if success:
                frame_counter += 1
                # Se passou 1 segundo, calcula o FPS e atualiza no dicionário para este ID
                if current_time - start_time >= 1.0:
                    fps = frame_counter / (current_time - start_time)
                    # Reinicia a contagem
                    frame_counter = 0
                    start_time = current_time
                # Redimensiona o frame
                frame = cv2.resize(frame, (1280, 720))
                
                fps_text = (f"FPS: {fps}")
                frame = cv2.putText(frame, fps_text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                
                with frame_lock:
                    global_frames[id] = frame
            else:
                cap.grab()


def draw_fps_in_frame(id, frame):
    """
    Cria um novo frame com o FPS desenhado, utilizando o valor global_fps para o ID fornecido.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Validação se o ID possui um FPS definido
    if id not in global_fps:
        print(f"ID {id} não encontrado em global_fps.")
        sleep(1)  # ou tratar de outra forma
        return None

    # Validação se o frame existe
    if id >= len(global_frames) or global_frames[id] is None:
        print(f"Frame inválido para o ID {id} em global_frames.")
        sleep(1)
        return None

    # Obter o valor do FPS e formatar o texto
    fps_value = global_fps.get(id, 0)
    fps_text = f"FPS: {fps_value:.2f}"

    # Cria uma cópia do frame original para não sobrescrever o frame original
    new_frame = global_frames[id].copy()

    # Desenha o texto no frame (posição, fonte, escala, cor, espessura e tipo de linha)
    new_frame = cv2.putText(new_frame, fps_text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

    # Armazena o novo frame globalmente
    frame = new_frame

    return frame


def generate_raw_camera(camera_id):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100, (cv2.IMWRITE_JPEG_OPTIMIZE), 1]
    

    while True:
        sleep(0.05)
        with frame_lock:
            frame = global_frames[camera_id]
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' + b'Aguardando o frame...\r\n')


@app.route('/video_raw<camera_id>')
def video_raw_camera_feed(camera_id):
    try:
        camera_id = int(camera_id)
        if 0 <= camera_id < len(camera_urls):
            #returnJson()
            
            return Response(generate_raw_camera(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return f"ID da câmera inválido: {camera_id}", 404
    except ValueError:
        return "O ID da câmera deve ser um inteiro.", 400


if __name__ == '__main__':
    # Inicializa threads de atualização de frames para cada câmera
    
    threads = []
    
    for idx, url in enumerate(camera_urls):
        thread = Thread(target=imageUpdater, kwargs={'id': idx, 'video_path': url, 'interval': 0.01})
        thread.start()
        threads.append(thread)
    
    # Inicializa o servidor Flask
    app.run(host='0.0.0.0', port=5000)
