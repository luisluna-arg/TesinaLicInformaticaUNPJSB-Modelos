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

    real_absolute_fft = 2.0 / number_of_datapoints * np.abs(complex_fft)

    result = list()
    for val in real_absolute_fft:
        result.append(float("{:.8f}".format(val)))

    arrayResult = np.array(result)

    print("arrayResult[0]", arrayResult[0], "arrayResult.shape", len(arrayResult), "number_of_datapoints", number_of_datapoints)
    print("arrayResult[0]", arrayResult[len(arrayResult)-1], "arrayResult.shape", len(arrayResult), "number_of_datapoints", number_of_datapoints)

    return arrayResult

def apply_fft(data):
    data_points = np.array(data, dtype=float)

    sample_count = data_points.shape[0]
    column_count = data_points.shape[1]
    indexes = range(0, column_count)
    result = np.empty((sample_count, column_count), float)
    print("indexes", indexes)
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

            # print("fft_array")
            # print(np.where(fft_array > 1))
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
