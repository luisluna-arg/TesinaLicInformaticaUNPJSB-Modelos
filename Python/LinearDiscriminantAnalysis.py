import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from load_files import load_files
from sklearn.utils import shuffle

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


n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


columns, data = load_files()

data = shuffle(data)

samples = list()
labels = list()

for currentData in data:
    sample = list()
    features = currentData[0: len(currentData) - 2]
    label = currentData[len(currentData) - 1]
    samples.append(features)
    labels.append(label)


# samples = normalization(np.array(samples))


total_count = len(samples)
testing_count = 10
training_count = total_count - testing_count

training_samples = samples[0: training_count - 1]
training_labels = labels[0: training_count - 1]
testing_samples = samples[training_count - 1: total_count]
testing_labels = labels[training_count - 1: total_count]


# def generate_data(n_samples, n_features):
#     """Generate random blob-ish data with noisy features.

#     This returns an array of input data with shape `(n_samples, n_features)`
#     and an array of `n_samples` target labels.

#     Only one feature contains discriminative information, the other features
#     contain only noise.
#     """
#     X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

#     # add non-discriminative features
#     if n_features > 1:
#         X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
#     return X, y

# X, y = generate_data(n_train, n_features)
X = training_samples
y = training_labels

clf1 = LinearDiscriminantAnalysis(
    solver="lsqr", shrinkage="auto").fit(X, y)
clf2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None).fit(X, y)
oa = OAS(store_precision=False, assume_centered=False)
clf3 = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa).fit(
    X, y
)

# X, y = generate_data(n_test, n_features)

def assert_porcent(prediction, testing_labels):
    correct = 0
    for idx, val in enumerate(prediction):
        correct += 1 if prediction[idx] == testing_labels[idx] else 0

    return round(correct / len(testing_labels) * 100, 2)

X = testing_samples
y = testing_labels

print("  Prediction", np.array(y))
prediction = clf1.predict(X)
print("predict_clf1", prediction, "acierto", assert_porcent(prediction, y), "%")
prediction = clf2.predict(X)
print("predict_clf2", prediction, "acierto", assert_porcent(prediction, y), "%")
prediction = clf2.predict(X)
print("predict_clf3", prediction, "acierto", assert_porcent(prediction, y), "%")
print("")
