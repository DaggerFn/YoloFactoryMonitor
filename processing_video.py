import cv2
from ultralytics import YOLO
from time import time, sleep
from threading import Lock
from numpy import zeros, uint8
from datetime import timedelta
from config import camera_urls, rois, roi_points_worker
from threading import Lock


# Inicializa os frames e frames anotados globais
global_frames = [None] * len(camera_urls)

#Frame para deteção e contagem de motor
generate_camera_motor_motor = [None] * len(camera_urls)

#Trheding Incializador
frame_lock = Lock()

#Frame para corte da area de contagem
global_cropped_frames = [None] * len(camera_urls)

#Frame para area de Status de Operação 
frames_worker = [None] * len(camera_urls)

#Frame com Deteções da Area de status de Operação
annotated_frames_worker = [None] * len(camera_urls)

# Para Contabilizar Quantidade de Motor
contador = {}
estado_anterior = {}
tempo_ultima_detecao = {}  # Dicionário para armazenar o tempo da última detecção


# Para contabilizar Estado da Operação
operacao = {}
operacao_anterior = {}
tempo_ultima_operacao = {}  # Dicionário para armazenar o tempo da última Operação
classes_operation = {}

# Tempo de cooldown em segundos
TEMPO_DE_COOLDOWN = 2  

lock = Lock()

tempo_ultima_detecao = {}  # Dicionário para armazenar o tempo da última detecção


#FPS  VAR

#Fps por id 
global_fps = {}  # Exemplo: {0: 15.23, 1: 12.34, ...}

# global_fps_frames: lista para armazenar os frames com o FPS desenhado
global_fps_frames = [None] * len(camera_urls)

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
    fps_text = None
    
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
                    fps_text = (f"FPS: {fps:.2f}")
                # Redimensiona o frame e adiciona FPS
                
                frame = cv2.resize(frame, (1280, 720))
                frame = cv2.putText(frame, fps_text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                
                with frame_lock:
                    global_frames[id] = frame        # Atualiza os frames cortados conforme suas funções
                crop_frames_by_rois()
                crop_frames_by_rois_worker()
                # Desenha o FPS no frame e atualiza o global_fps_frames
            else:
                cap.grab()


"""
def imageUpdater(id, video_path, interval):

    global global_frames, global_fps
    
    cap = cv2.VideoCapture(video_path)
    last_time = time()
    start_time = time()
    frame_counter = 0

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
                    global_fps[id] = fps
                    # Reinicia a contagem
                    frame_counter = 0
                    start_time = current_time
                # Redimensiona o frame
                frame = cv2.resize(frame, (480, 320))
                with frame_lock:
                    global_frames[id] = frame
                # Atualiza os frames cortados conforme suas funções
                crop_frames_by_rois()
                crop_frames_by_rois_worker()
                # Desenha o FPS no frame e atualiza o global_fps_frames
            else:
                cap.grab()

#Update imagens but no count fps

def imageUpdater(id, video_path, interval):
    global global_frames
    cap = cv2.VideoCapture(video_path)
    last_time = 0
    while True:
        current_time = time()
        if current_time - last_time >= interval:
            last_time = current_time
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (480, 320))  # Redimensiona o frame
                with frame_lock:
                    global_frames[id] = frame
                crop_frames_by_rois()
                crop_frames_by_rois_worker()
        else:
            cap.grab()
"""

def crop_frames_by_rois():
    """
    Realiza o corte nos frames armazenados em global_frames com base nas ROIs definidas.
    Atualiza a lista global_cropped_frames.
    """
    with frame_lock:
        for idx, frame in enumerate(global_frames):
            if frame is not None:
                roi_points = rois[idx]['points']
                mask = zeros(frame.shape[:2], dtype=uint8)
                cv2.fillPoly(mask, [roi_points], 255)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                x, y, w, h = cv2.boundingRect(roi_points)
                cropped_frame = masked_frame[y:y+h, x:x+w]
                global_cropped_frames[idx] = cropped_frame
            else:
                global_cropped_frames[idx] = None


def crop_frames_by_rois_worker():
    """
    Realiza o corte nos frames armazenados em global_frames com base nas ROIs definidas.
    Atualiza a lista global_cropped_frames.
    """
    with frame_lock:
        for idx, frame in enumerate(global_frames):
            if frame is not None:
                roi_points = roi_points_worker[idx]['point']
                mask = zeros(frame.shape[:2], dtype=uint8)
                cv2.fillPoly(mask, [roi_points], 255)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                x, y, w, h = cv2.boundingRect(roi_points)
                cropped_frame_worker = masked_frame[y:y+h, x:x+w]
                frames_worker[idx] = cropped_frame_worker
            else:
                frames_worker[idx] = None


def varReturn():
    global contador,operacao
    
    with lock:  # Bloquear ao acessar a variável global
        if len(contador) and len(operacao) == len(camera_urls):
            return contador, operacao

def logInfo(id):
    
        #print("###################################")
        #print('contador e \n',contador)
        #print(f"[{id}] Motor contado! Total: {contador[id]['Quantidade']}")
        #print('No id', [id], operacao[id])
        #print("###################################")
        #print('Tamanho de contador',len(contador))
        #print('Tamanho de operacação',len(operacao))
        None
    

def count_motor(id):
    global global_frames, global_cropped_frames, generate_camera_motor_motor
    global contador, estado_anterior, tempo_ultima_detecao  

    #pt model
    #model = YOLO(r'/home/sim/code/models/modelo_linha/linha_11m.pt').to('cuda')
    
    #OpenVino model
    model = YOLO(r'models\linha_11m.pt')#.to('cpu')
    
    while True:
        start = time()

        try:
            with frame_lock:
                frame = global_cropped_frames[id]

            if frame is not None:
                results = model.predict(frame, augment=True, visualize=False, verbose=False, conf=0.8, iou=0.1, imgsz=640)

                detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
                tem_motor = any('motor' in cls.lower() for cls in detected_classes)    

                if id not in contador:
                    contador[id] = {'Quantidade': 0}
                if id not in estado_anterior:
                    estado_anterior[id] = 0
                if id not in tempo_ultima_detecao:
                    tempo_ultima_detecao[id] = 0  

                # Verifica se o motor apareceu e antes não estava presente
                if tem_motor:
                    tempo_atual = time()
                    tempo_decorrido = tempo_atual - tempo_ultima_detecao[id]

                    # Incrementa apenas se passou o cooldown e antes não estava presente
                    if estado_anterior[id] == 0 and tempo_decorrido > TEMPO_DE_COOLDOWN:
                        contador[id]['Quantidade'] += 1
                        
                        # Atualiza o tempo da última detecção
                        tempo_ultima_detecao[id] = tempo_atual

                    # Atualiza estado para indicar que o motor está presente
                    estado_anterior[id] = 1  

                    
                    
                else:
                    # Se o motor sumiu, atualiza o estado para 0
                    estado_anterior[id] = 0  

                logInfo(id)
                    
                with frame_lock:
                    generate_camera_motor_motor[id] = results[0].plot(conf=True, labels=True, line_width=1)

            else:
                with frame_lock:
                    generate_camera_motor_motor[id] = zeros((320, 480, 3), dtype=uint8)  

        except Exception as e:
            print(e)
            pass



def count_operation(id):
    global global_frames
    global frames_worker
    global annotated_frames_worker
    global rois
    global classes_operation, operacao, operacao_anterior, tempo_ultima_operacao
    
    #pt model
    model = YOLO(r'models\linha_11m.pt')
    
    #OpenVino model
    #model = YOLO('/home/sim/code/models/linha_11m_openvino_model/')#.to('cpu')
    #model = YOLO(r'linha_11m_openvino_model/')#.to('cuda')
    
    
    while True:
        start = time()
        #start = time.time()
        try:
            with frame_lock:
                #frame = global_cropped_frames[id]
                frame = frames_worker[id]
            if frame is not None:
                annotated_frame = frame
                
                
                #OpenVino
                #results = model.predict(frame, augment=True, task="detect", visualize=False, verbose=False, conf=0.7, iou=0.5)#,imgsz=544)
                
                #GPU
                results = model.predict(frame, augment=True, visualize=False, verbose=False, conf=0.6, iou=0.1, imgsz=544)
                

                classes_operation = [model.names[int(cls)] for cls in results[0].boxes.cls]
                annotated_frame = results[0].plot(conf=True, labels=True, line_width=1)
                
                tem_operacao = any('motor' and 'hand'in cls.lower() for cls in classes_operation)    
                
                if id not in operacao:
                    operacao[id] = {'Operação': 0}
                if id not in operacao_anterior:
                    operacao_anterior[id] = 0
                if id not in tempo_ultima_operacao:
                    tempo_ultima_operacao[id] = 0  
                
                if tem_operacao:
                    operacao[id]['Operação'] = 'Operando'
                else:
                    operacao[id]['Operação'] = 'Parado'

                with frame_lock:
                    annotated_frames_worker[id] = annotated_frame
            else:
                with frame_lock:
                    annotated_frames_worker[id] = zeros((320, 480, 3), dtype=uint8)  # Adicione um frame vazio se não houver frame

        except Exception as e:
            print(e)
            pass
        inference_time = time() - start
        # print(f'{inference_time=}')




def generate_raw_camera(camera_id):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

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


def generate_camera_motor(camera_id):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 1]

    while True:
        sleep(0.05)
        with frame_lock:
            frame = generate_camera_motor_motor[camera_id]
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' + b'Waiting for the frame...\r\n')


def generate_cropped_frames(camera_id):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 1]

    while True:
        sleep(0.05)
        with frame_lock:
            frame = annotated_frames_worker[camera_id]
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' + b'Waiting for the cropped frame...\r\n')

