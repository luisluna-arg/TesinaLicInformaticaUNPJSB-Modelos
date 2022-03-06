import math
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
# from load_files import load_files
from load_csv_files import read_file

###############################################################
###############################################################
###############################################################

def normalization(data):
    sample_count = data.shape[0]
    column_count = data.shape[1]
    indexes = range(0, column_count - 1)
    result = np.empty((sample_count, column_count), float)
    print("shape", result.shape)
    for ix in indexes:
        temp_array = data[0:sample_count, ix:ix+1].flatten()
        
        max = np.max(temp_array)
        min = np.min(temp_array)

        for normal_ix, sample in enumerate(temp_array):
            partial_res = (sample - min) / (max - min)
            result[normal_ix, ix] = partial_res

    return np.asarray(result)

###############################################################
###############################################################
###############################################################

columns, samples = read_file()
loop_count = 30
tests = list()
scores = list()



data_np = np.array(samples)
np.random.shuffle(data_np)

# Separar
row_count = data_np.shape[0]
column_count = data_np.shape[1]

samples = data_np[0:row_count, 1:column_count-1]
labels = data_np[0:row_count, 0:1]

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

rng = np.random.RandomState(1)
clf = MultinomialNB()
clf.fit(X, Y)

single_score = clf.score(X, Y)
scores.append(single_score)


predictions = clf.predict(test_samples).flatten()

asserts = 0
for ix, item in enumerate(predictions):
    if test_labels[ix] == item:
        asserts += 1

if asserts == 0:
    # tests.append(0)
    print("test_size", test_size, "| Precisión", 0, "| Score", single_score)
else:
    precision = asserts / predictions.shape[0] * 100
    # tests.append(precision)
    print("test_size", test_size, "| Precisión", precision, "| Score", single_score)

# print("aciertos {0} de {1}, {2}%".format(asserts, label_count, correct_porcent))
# print('')
