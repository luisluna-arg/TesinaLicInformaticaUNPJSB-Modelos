import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from load_files import load_files

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

columns, data = load_files()

# samples_sparse = coo_matrix(data)

data = shuffle(data)

samples = list()
labels = list()

for currentData in data: 
    sample = list()
    features = currentData[0: len(currentData) - 2]
    label = currentData[len(currentData) - 1]
    samples.append(features)
    labels.append(label)


samples = normalization(np.array(samples))

print("samples")
print(samples)






total_count = len(samples)
testing_count = 10
training_count = total_count - testing_count

training_samples = samples[0 : training_count - 1] 
training_labels = labels[0 : training_count - 1] 
testing_samples = samples[training_count - 1 : total_count] 
testing_labels = labels[training_count - 1 : total_count] 

label_count = len(testing_labels)

rng = np.random.RandomState(1)
clf = MultinomialNB()
clf.fit(training_samples, training_labels)

print('')
print('training_samples')
print(training_samples[0])
print('')
print('testing_labels')
print(testing_labels)
print('')
print('predictions')
prediction = clf.predict(testing_samples).flatten()
print(prediction)
print('')

correct = 0
for idx, val in enumerate(prediction):
    correct += 1 if prediction[idx] == testing_labels[idx] else 0

correct_porcent = round(correct / len(testing_labels) * 100, 2)

print("aciertos {0} de {1}, {2}%".format(correct, label_count, correct_porcent))
print('')
