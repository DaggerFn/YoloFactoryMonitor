from threading import Lock
from time import time, sleep
from datetime import datetime


'''
quantidade = {0 :{'Quantidade': 1},
       1 :{'Quantidade': 0},
       }       

status = {0 :{'Operacao': 0},
       1 :{'Operacao': 1},
       }       
'''

lock_var = Lock()
postos = {}

def updateDateAndTime():
    now = datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')#, now.strftime('%H:%M:%S')


def makeJson(varReturn):
    global postos
    
    while True:
        try:
            contador, operacao = varReturn()
            
            for i in range(len(operacao)):
                
                index_posto = f'Posto{i + 1}'
                
                if index_posto not in postos:
                
                    postos[index_posto] = {
                        'Data': updateDateAndTime(),
                        'Status': operacao[i]['Operação'],
                        'Quantidade': contador[i]['Quantidade'],
                    }
                    print('OK in JSON')
                
                
                else:
                    #print('\n')
                    #print('Stoping Looking Variables')
                    sleep(2)
                    postos[index_posto] = {
                        'Data': updateDateAndTime(),
                        'Status': operacao[i]['Operação'],
                        'Quantidade': contador[i]['Quantidade'],
                    }
        except:
            print('______________________\n')
            print('Waitig lenght correct...')        
            print('______________________\n')
            sleep(6)


def updateAPI():    
    global postos
    
    #with lock_var:    
    #    return postos
    return postos
    