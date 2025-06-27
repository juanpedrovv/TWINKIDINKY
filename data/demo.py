import os
import dill
import wget
import pandas as pd
from data.patient_data import TabularPatientBase

SEQ_TRIAL_PATIENT_URL = 'https://storage.googleapis.com/pytrial/seq_patient_nct00174655.zip'


def load_trial_patient_sequence(input_dir=None):
    '''
    Es la responsable de leer los datos "crudos" del ensayo.

    ¿Qué hace?: Carga varios archivos .pkl y .csv que contienen la información del ensayo clínico. Entre ellos, 
                los más importantes para nosotros ahora son:

            - visit.pkl: Contiene la secuencia de visitas de cada paciente. Esta es la estructura de datos principal 
                         que representa las trayectorias de los pacientes.

            - voc.pkl: Contiene los "vocabularios", que son diccionarios que mapean cada evento (p. ej., un medicamento específico) 
                       a un número entero. Piensa en esto como un "traductor" de eventos a un formato que el modelo puede entender.

    Parameters
    ----------
    input_dir: str
        The folder that stores the demo data. If None, we will download the demo data and save it
        to './demo_data/demo_patient_sequence/trial'. Make sure to remove this folder if it is empty.    
    '''

    # Para descomprimir la data que esta en el .zip en la nube ('https://storage.googleapis.com/pytrial/seq_patient_nct00174655.zip')
    if input_dir is None:
        input_dir = './demo_data/demo_patient_sequence/trial'
    
        # if not os.path.exists(input_dir):
        #     os.makedirs(input_dir)
        #     url = SEQ_TRIAL_PATIENT_URL
        #     filename = wget.download(url, out=input_dir)
        #     # unzip filename
        #     import zipfile
        #     with zipfile.ZipFile(filename, 'r') as zip_ref:
        #         zip_ref.extractall(input_dir)
        #     print(f'\n Download trial patient sequence data to {input_dir}.')


    print("#"*5+'Demo Data Folder'+"#"*5)
    print(os.listdir(input_dir))
    print("#"*20)

    # 1. Carga de archivos
    visit = dill.load(open(os.path.join(input_dir,'visit.pkl'), 'rb')) # Matriz. Cada lista interior es un paciente y dentro de cada paciente hay una lista de visitas. Cada visita es una lista que contiene los eventos ocurridos.
    vocs = dill.load(open(os.path.join(input_dir,'voc.pkl'), 'rb')) #Diccionario traductor. convierten los nombres de los eventos (como "dolor de cabeza") en números enteros. {'dolor de cabeza': 1, 'náuseas': 2, ...}
    feature = pd.read_csv(os.path.join(input_dir, 'feature.csv')) #Características estáticas de los pacientes (peso, altura, etc.).
    v_stage = dill.load(open(os.path.join(input_dir,'visit_stage.pkl'), 'rb')) #Representa las etapas de la enfermedad de un paciente en cada visita.Por ejemplo, si un paciente tiene 5 visitas registradas en visit.pkl, visit_stage contendría una secuencia de 5 elementos indicando el estadio de la enfermedad en cada una de esas visitas.
    orders = list(vocs.keys())
    
    # 2. Procesamiento de Características Estáticas (de feature.csv)
    label_relapse = feature['num relapse']
    label_mortality = feature['death'].values
    x = feature.drop(['num relapse','death','RUSUBJID'], axis=1)
    x['weight'] = x['weight'].replace({'>= 125':'125'}).astype(float)

    # 3. Procesamiento de Características Estáticas (de feature.csv)
    tabx = TabularPatientBase(x)

    x = tabx.df.values # get processed patient features in matrix form
    return {
        'feature':x,
        'visit':visit,
        'voc':vocs,
        'order':orders,
        'visit_stage':v_stage,
        'relapse':label_relapse,
        'mortality':label_mortality,
    } 