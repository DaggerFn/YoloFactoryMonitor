from flask import Flask, render_template
from flask import Flask, Response, jsonify
from flask_cors import CORS  # Importa a biblioteca CORS
from utils import load_yolo_model, generate_frames, get_tracking_info

app = Flask(__name__)
CORS(app)  # Adiciona CORS ao seu app Flask

# Configurações
MODEL_PATH = 'best0.pt'
CAMERA_URL = 'http://10.1.60.155:4000/video_feed'

# Carregar o modelo YOLOv10
model = load_yolo_model(MODEL_PATH)

@app.route('/video_feed')
def video_feed():
    """Rota do Flask para transmitir o vídeo processado."""
    return Response(generate_frames(model, CAMERA_URL),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tracking_info', methods=['GET'])
def tracking_info():
    """Rota do Flask para obter as informações de rastreamento no formato JSON."""
    info = get_tracking_info()
    return jsonify(info)

@app.route('/tracking')
def tracking():
    """Rota do Flask para servir a página de tracking info."""
    return render_template('tracking.html')

def main():
    """Função principal para iniciar o servidor Flask."""
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    main()
