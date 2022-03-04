from json import load
import math
from operator import index
from unittest import result
from sklearn import tree
# from load_files import load_files
from load_csv_files import read_file
import numpy as np


def normalization(data):
    label_column_ix = 0
    
    sample_count = data.shape[0]
    column_count = data.shape[1]
    indexes = range(0, column_count - 1)
    result = np.empty((sample_count, column_count), float)
    print("shape", result.shape)
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


columns, data = read_file()
loop_count = 30
tests = list()
scores = list()

for ix in range(loop_count):

    # Convertir a arreglo numpy
    data_np = np.array(data)
    np.random.shuffle(data_np)

    # Separar
    row_count = data_np.shape[0]
    column_count = data_np.shape[1]

    samples = data_np[0:row_count, 1:column_count-1]
    labels = data_np[0:row_count, 0:1]

    # Normalizar muestras 
    # samples = normalization(samples)

    test_size = math.floor(row_count * 0.3)
    train_count = row_count - test_size
    test_count = row_count - train_count
    feature_count = samples.shape[1]

    # Entrenamiento y etiquetas
    train_samples = samples[0:train_count, 0:feature_count]
    train_labels = labels[0:train_count]

    test_samples = samples[train_count:row_count, 0:feature_count]
    test_labels = labels[train_count:row_count].flatten()

    # Entrenamiento
    X = train_samples
    Y = train_labels

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    single_score = clf.score(X, Y)
    scores.append(single_score)

    # Testing
    # print("Cant.", row_count)
    # print("  Esperado", test_labels)
    predictions = clf.predict(test_samples)
    # print("Predicción", predictions)

    asserts = 0
    for ix, item in enumerate(predictions):
        if test_labels[ix] == item:
            asserts += 1

    if asserts == 0:
        tests.append(0)
        print("test_size", test_size, "| Precisión", 0, "| Score", single_score)
    else:
        precision = asserts / predictions.shape[0] * 100
        tests.append(precision)
        print("test_size", test_size, "| Precisión", precision, "| Score", single_score)

print("Promedios | precision: ", np.average(tests), "Score", np.round(np.average(scores), 2))