#!/usr/bin/python3
import cv2
from time import time, sleep
from multiprocessing import Process, Manager, Lock
from flask import Flask, Response
import numpy as np
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Ou logging.CRITICAL para menos logs

# Define a lista de URLs das câmeras
camera_urls = [
    "rtsp://admin:fabrica1@10.1.30.1:554/1/1",  # Indice 0/1
]

app = Flask(__name__)

def imageUpdater(id, video_path, interval, global_frames, frame_lock):
    """
    Atualiza o frame da câmera, calcula o FPS e desenha-o no frame.
    """
    cap = cv2.VideoCapture(video_path)
    last_time = time()
    start_time = time()
    frame_counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps_text = None
    
    while True:
        current_time = time()
        if current_time - last_time >= interval:
            last_time = current_time
            success, frame = cap.read()
            if success:
                frame_counter += 1
                if current_time - start_time >= 1.0:
                    fps = frame_counter / (current_time - start_time)
                    frame_counter = 0
                    start_time = current_time
                    fps_text = (f"FPS: {fps:.2f}")

                frame = cv2.resize(frame, (1920, 1080))
                frame = cv2.putText(frame, fps_text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                
                with frame_lock:
                    global_frames[id] = frame
            else:
                cap.grab()


def generate_raw_camera(camera_id, global_frames, frame_lock):
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
            return Response(generate_raw_camera(camera_id, global_frames, frame_lock), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return f"ID da câmera inválido: {camera_id}", 404
    except ValueError:
        return "O ID da câmera deve ser um inteiro.", 400


if __name__ == '__main__':
    manager = Manager()
    global_frames = manager.list([np.zeros((320, 480, 3), dtype=np.uint8)] * len(camera_urls))
    frame_lock = manager.Lock()

    processes = []
    
    # Inicializa processos de atualização de frames para cada câmera
    for idx, url in enumerate(camera_urls):
        process = Process(target=imageUpdater, kwargs={'id': idx, 'video_path': url, 'interval': 0.01, 
                                                       'global_frames': global_frames, 'frame_lock': frame_lock})
        process.start()
        processes.append(process)
    
    # Inicializa o servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=False)

    # Garante que os processos terminem corretamente
    for process in processes:
        process.join()
