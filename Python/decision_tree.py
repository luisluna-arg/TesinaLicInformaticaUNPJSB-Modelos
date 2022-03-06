from json import load
import math
from operator import index
from unittest import result
from sklearn import tree
# from load_files import load_files
from load_csv_files import read_file
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


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
train_scores = list()
test_scores = list()
rmses = list()
accs = list()

data_np = np.array(data)
row_count = data_np.shape[0]
column_count = data_np.shape[1]
test_size = math.floor(row_count * 0.3)

print("")
print("Size:", test_size)
print("")

for ix in range(loop_count):

    # Convertir a arreglo numpy

    np.random.shuffle(data_np)

    # Separar
    samples = data_np[0:row_count, 1:column_count-1]
    labels = data_np[0:row_count, 0:1]

    # Normalizar muestras
    # samples = normalization(samples)

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

    clf = tree.DecisionTreeClassifier(
        criterion='entropy',
        min_samples_split=20,
        min_samples_leaf=4,
        # max_depth=4,
        # class_weight={1: 3.5}
        # max_depth=16
        # , random_state=123
    )
    clf = clf.fit(X, Y)

    train_score = clf.score(X, Y)
    test_score = clf.score(test_samples, test_labels)
    train_scores.append(train_score)
    test_scores.append(test_score)

    predictions = clf.predict(test_samples)
    # predictions = clf.predict_log_proba(test_samples)

    rmse = mean_squared_error(
        y_true=test_labels,
        y_pred=predictions,
        squared=False
    )
    rmses.append(rmse)

    accuracy = 100 * accuracy_score(
        y_true=test_labels,
        y_pred=predictions,
        normalize=True
    )
    accs.append(accuracy)

    asserts = 0
    for ix, item in enumerate(predictions):
        if test_labels[ix] == item:
            asserts += 1

    if asserts == 0:
        tests.append(0)
        print(
            "Prec.", 0,
            "| Score Train:", np.round(train_score, 2),
            "| Score Test:", np.round(test_score, 2),
            "| RMSE:", np.round(rmse, 2),
            "| Acc.:", np.round(accuracy, 2),
            "| Acc.:", np.round(accuracy, 2),
        )
    else:
        precision = asserts / predictions.shape[0] * 100
        tests.append(precision)
        print(
            "Prec.:", np.round(precision, 2),
            "| Score Train:", np.round(train_score, 2),
            "| Score Test:", np.round(test_score, 2),
            "| RMSE:", np.round(rmse, 2),
            "| Acc.:", np.round(accuracy, 2)
        )

print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("Proms. "
        "| Prec.:", np.round(np.average(tests), 2),
        "| Train Score:", np.round(np.average(train_scores), 2),
        "| Test Score:", np.round(np.average(test_scores), 2),
        "| RMSE:", np.round(np.average(rmses), 2),
        "| ACC:", np.round(np.average(accs), 2)
)
print("")