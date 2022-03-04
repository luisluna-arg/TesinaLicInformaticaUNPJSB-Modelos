from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from load_files import  load_files
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

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

X = samples
y = labels
y = tf.keras.utils.to_categorical(y) #converting output to one-hot vector
ss = StandardScaler() #standardizing the data
X = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y) #splitting dataset into 2 halves



model11 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512//2, activation='tanh'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512//4, activation='tanh'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512//8, activation='tanh'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(5, activation='softmax')
])
model11.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])
hist11 = model11.fit(X_train, y_train, epochs=350,  validation_data=(X_test,y_test), verbose=2)


loss11, acc11, mse11 = model11.evaluate(X_test, y_test)
print(f"Loss is {loss11},\nAccuracy is {acc11 * 100},\nMSE is {mse11}")

# print(labels.distinct|)