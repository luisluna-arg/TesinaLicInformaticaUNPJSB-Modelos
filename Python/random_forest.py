from locale import normalize
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from scipy.special import softmax, log_softmax
import json
import numpy as np
import math

###############################################################
###############################################################
###############################################################

arribaIx = 0
abajoIx = 1
izquierdaIx = 2
derechaIx = 3

def importFile(samples, file_path, direction):
    f = open(file_path)
    data = json.load(f)
    f.close()

    for i in data: 
        sample = list()
        sample.append(i['eegPower']['delta'])
        sample.append(i['eegPower']['theta'])
        sample.append(i['eegPower']['lowAlpha'])
        sample.append(i['eegPower']['highAlpha'])
        sample.append(i['eegPower']['lowBeta'])
        sample.append(i['eegPower']['highBeta'])
        sample.append(i['eegPower']['lowGamma'])
        sample.append(i['eegPower']['highGamma'])
        sample.append(direction)
        samples.append(sample)
        

columns = list()

columns.append('delta')
columns.append('theta')
columns.append('lowAlpha')
columns.append('highAlpha')
columns.append('lowBeta')
columns.append('highBeta')
columns.append('lowGamma')
columns.append('highGamma')
columns.append('direction')


###############################################################
###############################################################
###############################################################


data = list()

abajoIx = 1
arribaIx = 2
izquierdaIx = 3
derechaIx = 4

# base_path = 'data/2022.2.22/'
# base_path = 'data/Full/'
base_path = 'data/generated/'

importFile(data, base_path + 'ABAJO.json', abajoIx)
importFile(data, base_path + 'ARRIBA.json', arribaIx)
importFile(data, base_path + 'IZQUIERDA.json', izquierdaIx)
importFile(data, base_path + 'DERECHA.json', derechaIx)




# import csv
# # open the file in the write mode
# f = open(base_path + '/data.csv', 'w')

# # create the csv writer
# writer = csv.writer(f)

# columns_str = ','.join(columns)

# # write a row to the csv file
# writer.writerow(columns)

# writer.writerows(data)

# # for row in data:
# #     int_str = [str(int) for int in row]
# #     writer.writerow(int_str)

# # close the file
# f.close()

# print('file created')


def normalize(samples_to_normalize):
    # scaler = preprocessing.StandardScaler().fit(samples_to_normalize)
    # return scaler.transform(samples_to_normalize)
    tansposed_features = np.transpose(samples_to_normalize)
    normalized_samples = list()
    for idx, current_sample in enumerate(tansposed_features):
        input_max = np.max(current_sample)
        input_min = np.min(current_sample)
        if (input_max - input_min != 0 and idx < 8):
            print("currentSample")
            print(current_sample)
            substracted = current_sample - input_min
            divided = substracted / (input_max  - input_min)
            normalized_samples.append(divided)
        else:
            normalized_samples.append(current_sample)
    
    return np.transpose(normalized_samples)
    




data = shuffle(data)

print ('data[0][0]', data[0][0])
print ('data[0][len(columns)]', data[0][len(columns) - 1])


# transposedData = np.transpose(data)

# normalized_data = list()
# for idx, data_row in enumerate(transposedData):
#     # normalized_data.append(softmax(data_row) if (idx < len(columns) - 1) else data_row)
#     normalized_data.append(np.log(data_row) if (idx < len(columns) - 1) else data_row)
#     # normalized_data.append(np.log(data_row) if (idx < len(columns) - 1) else data_row)

# data = np.transpose(normalized_data)

data = normalize(data)

print ('data[0][0]', data[0][0])
print ('data[0][len(columns)]', data[0][len(columns) - 1])











#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
print('Libraries Imported')


column_count = len(columns)

dataset = pd.DataFrame(data=data, columns=columns)

# #Creating Dataset and including the first row by setting no header as input
# dataset = pd.read_csv('iris.data.csv', header = None)
# #Renaming the columns
# dataset.columns = ['sepal length in cm', 'sepal width in cm','petal length in cm','petal width in cm','direction']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head(15))


#Creating the dependent variable class
factor = pd.factorize(dataset['direction'])
dataset.direction = factor[0]
definitions = factor[1]
print(dataset.direction.head())
print(definitions)



#Splitting the data into independent and dependent variables
dependant_variable_id = column_count - 1
x = dataset.iloc[:,0:dependant_variable_id].values
y = dataset.iloc[:,dependant_variable_id].values
print('The independent features set: ')
print(x[:dependant_variable_id,:])
# print(len(x))
# print(x)
print('The dependent variable: ')
print(y[:dependant_variable_id])
# print(len(y))
# print(y)


# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 21)


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
reversefactor = dict(zip(range(len(columns)),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual Direction'], colnames=['Predicted Direction'])

correct = confusion_matrix.get(0,0) + confusion_matrix.get(1,1) + confusion_matrix.get(2,2) + confusion_matrix.get(3,3)

suma_diagonal = sum(np.diagonal(confusion_matrix))

print("suma_diagonal", suma_diagonal)
print("len(data)", len(data))

print("Precision 0%".format(suma_diagonal / len(data) * 100))