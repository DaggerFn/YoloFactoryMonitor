from video_utils import get_motor_sensor_value
from datetime import datetime

# Variáveis globais
sensorValue = 0
sensorValueAnterior = 0
qtdMotor = 0
data = datetime.now()
formatado = data.strftime('%Y-%m-%d %H:%M:%S')

def time_str_to_milliseconds(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    milliseconds = (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + time_obj.microsecond // 1000
    return milliseconds

# Função para incrementar a quantidade de motores
def incrementMotor():
    global qtdMotor, sensorValue, sensorValueAnterior
    
    # Verifica se o valor mudou de 0 para 1 (indicando a presença de um motor)
    if sensorValue == 1 and sensorValueAnterior == 0:
        qtdMotor += 1

# Main - Função para rastrear o valor atual e o anterior do sensor
def get_tracking_info():
    global sensorValue, sensorValueAnterior, data

    # Atualiza o valor anterior antes de mudar o valor atual
    sensorValueAnterior = sensorValue
    
    # Obtém o novo valor do sensor
    sensorValue = get_motor_sensor_value()

    # Incrementa o número de motores se necessário
    incrementMotor()

    print(data)
    
    # Retorna as informações em formato de dicionário
    return {
        'Sensor': sensorValue,
        'Sensor Anterior': sensorValueAnterior,
        'Quantidade de Motor': qtdMotor,
        'data': formatado
    }
    