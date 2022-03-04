from ast import For
import csv
import numpy as np
from scipy.fftpack import fft 

# csvFilePath = 'data/other_datasets/user_a.csv'
csvFilePath = 'data/csv/arriba_abajo_derecha_izquierda4.csv'

label_column_ix = 0

def normalization(data):
    sample_count = data.shape[0]
    column_count = data.shape[1]
    indexes = range(0, column_count - 1)
    result = np.empty((sample_count, column_count), float)

    for ix in indexes:
        temp_array = data[0:sample_count, ix:ix+1].flatten()
        
        max = np.max(temp_array)
        min = np.min(temp_array)

        if (ix == label_column_ix):
            # Ignorar las labels
            for normal_ix, sample in enumerate(temp_array):
                result[normal_ix, ix] = sample
        else: 
            for normal_ix, sample in enumerate(temp_array):
                partial_res = (sample - min) / (max - min)
                result[normal_ix, ix] = partial_res

    return result

def load_files():
    json_array = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csv_reader = csv.DictReader(csvf) 

        columns = csv_reader.fieldnames

        #convert each csv row into python dict
        for row in csv_reader: 
            #add this python dict to json array
            float_values = list()
            for value in list(row.values()):
                float_values.append(float(value))
            
            json_array.append(float_values)

    return columns, json_array

def fix_fft(data_points):
    number_of_datapoints = len(data_points)

    complex_fft = fft(data_points)
    real_absolute_fft = 2.0 / number_of_datapoints * \
                        np.abs(complex_fft[:number_of_datapoints//2])

    result = list()
    for val in real_absolute_fft:
        result.append(float("{:.8f}".format(val)))

    return np.array(result)

def apply_fft(data):
    data_points = np.array(data, dtype=float)

    sample_count = data_points.shape[0]
    column_count = data_points.shape[1]
    indexes = range(0, column_count - 1)
    result = np.empty((sample_count, column_count), float)
    for ix in indexes:
        temp_array = data[0:sample_count, ix:ix+1].flatten()

        if (ix == label_column_ix):
            # Ignorar las labels/clase
            # print ("ix", ix)
            # print ("temp_array", temp_array)
            for normal_ix, sample in enumerate(temp_array):
                result[normal_ix, ix] = sample
        else: 
            fft_array = fix_fft(temp_array)
            # print ("ix", ix)
            # print ("temp_array", temp_array)
            # print ("fft_array", fft_array)
            for normal_ix, sample in enumerate(fft_array):
                result[normal_ix, ix] = sample

    return result























def read_file():
    columnas, data = load_files()

    data = normalization(np.array(data))

    restored = apply_fft(data)

    # print(restored[0])

    return columnas, restored




# rows = load_files('data/other_datasets/user_a.csv')
# print(rows[0])



    # #convert python jsonArray to JSON String and write to file
    # with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
    #     jsonString = json.dumps(jsonArray, indent=4)
    #     jsonf.write(jsonString)

# def importFile(samples, file_path, direction):
#     csvFilePath = r'data/other_datasets/user_a.csv'
#     json_array = csv_to_json(csvFilePath)

#     for i in data: 
#         sample = list()
#         sample.append(i['eegPower']['delta'])
#         sample.append(i['eegPower']['theta'])
#         sample.append(i['eegPower']['lowAlpha'])
#         sample.append(i['eegPower']['highAlpha'])
#         sample.append(i['eegPower']['lowBeta'])
#         sample.append(i['eegPower']['highBeta'])
#         sample.append(i['eegPower']['lowGamma'])
#         sample.append(i['eegPower']['highGamma'])
#         sample.append(direction)
#         samples.append(sample)
        
# def load_files():
#     columns = list()

#     columns.append('delta')
#     columns.append('theta')
#     columns.append('lowAlpha')
#     columns.append('highAlpha')
#     columns.append('lowBeta')
#     columns.append('highBeta')
#     columns.append('lowGamma')
#     columns.append('highGamma')
#     columns.append('direction')

#     ###############################################################
#     ###############################################################
#     ###############################################################

#     data = list()

#     abajoIx = 1
#     arribaIx = 2
#     izquierdaIx = 3
#     derechaIx = 4

#     # base_path = 'data/2022.2.22/'
#     base_path = 'data/Full/'
#     # base_path = 'data/generated/'

#     importFile(data, base_path + 'ABAJO.json', abajoIx)
#     importFile(data, base_path + 'ARRIBA.json', arribaIx)
#     importFile(data, base_path + 'IZQUIERDA.json', izquierdaIx)
#     importFile(data, base_path + 'DERECHA.json', derechaIx)

#     return columns, data
