import numpy as np


# Define a lista de URLs das câmeras
camera_urls = [
    "rtsp://admin:fabrica1@192.168.0.131:554/1/2",  # Indice 0/1
    "rtsp://admin:fabrica1@192.168.0.132:554/1/2",  # Indice 0/2
    "rtsp://admin:fabrica1@192.168.0.133:554/1/2",  # Indice 0/3
    "rtsp://admin:fabrica1@192.168.0.134:554/1/2",  # Indice 0/4
    "rtsp://admin:fabrica1@192.168.0.135:554/1/2",  # Indice 0/5
    "rtsp://admin:fabrica1@192.168.0.136:554/1/2",  # Indice 0/6
]


rois = [
    {'points': np.array([[290, 120], [440, 120], [445, 275], [290, 275]], dtype=np.int32), 'color': (255, 0, 0)},
    {'points': np.array([[351, 115], [480, 115], [480, 320], [351, 320]], dtype=np.int32), 'color': (255, 0, 0)},
    {'points': np.array([[330, 135], [370, 150], [370, 270], [330, 285]], dtype=np.int32), 'color': (255, 0, 0)},
    {'points': np.array([[220, 220], [270, 220], [270, 320], [220, 320]], dtype=np.int32), 'color': (0, 255, 0)},
    {'points': np.array([[430, 80], [480, 54], [480, 140], [430, 180]], dtype=np.int32), 'color': (255, 0, 0)},
    {'points': np.array([[330, 135], [370, 150], [370, 270], [330, 285]], dtype=np.int32), 'color': (255, 0, 0)},
]


roi_points_worker = [
    {'point': np.array([[0, 115], [300, 115], [315, 320], [0, 320]], dtype=np.int32), 'color': (255, 0, 0)},
    {'point': np.array([[0, 115], [300, 115], [315, 320], [0, 320]], dtype=np.int32), 'color': (255, 0, 0)},
    {'point': np.array([[30, 135], [305, 150], [305, 270], [30, 285]], dtype=np.int32), 'color': (255, 0, 0)},
    {'point': np.array([[25, 180], [290, 220], [290, 320], [25, 320]], dtype=np.int32), 'color': (0, 255, 0)},
    {'point': np.array([[200, 190], [405, 60], [450, 155], [320, 320]], dtype=np.int32), 'color': (255, 0, 0)},
    {'point': np.array([[90, 227], [330, 140], [370, 320], [160, 320]], dtype=np.int32), 'color': (0, 255, 0)},
]



    #{'points': np.array([[220, 180], [270, 220], [270, 320], [220, 320]], dtype=np.int32), 'color': (0, 255, 0)},
    #{'points': np.array([[200, 190], [405, 60], [450, 155], [320, 320]], dtype=np.int32), 'color': (255, 0, 0)},
    #{'points': np.array([[90, 227], [330, 140], [370, 320], [160, 320]], dtype=np.int32), 'color': (0, 255, 0)},
