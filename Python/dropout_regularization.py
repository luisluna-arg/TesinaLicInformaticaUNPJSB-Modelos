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



import tensorflow as tf
model7 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape),
    tf.keras.layers.Dropout(0.5), #dropout with 50% rate
    tf.keras.layers.Dense(512//2, activation='tanh'),
    
    tf.keras.layers.Dense(512//4, activation='tanh'),
    tf.keras.layers.Dense(512//8, activation='tanh'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model7.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])
hist7 = model7.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test), verbose=False)


loss7, acc7, mse7 = model7.evaluate(X_test, y_test)
print(f"Loss is {loss7},\nAccuracy is {acc7 * 100},\nMSE is {mse7}")




model8 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512//2, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512//4, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512//8, activation='tanh'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model8.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])
hist8 = model8.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test), verbose=False)

loss8, acc8, mse8 = model8.evaluate(X_test, y_test)
print(f"Loss is {loss8},\nAccuracy is {acc8 * 100},\nMSE is {mse8}")






plt.figure(figsize=(15,8))
plt.plot(hist7.history['loss'], label = 'loss')
plt.plot(hist7.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



plt.figure(figsize=(15,8))
plt.plot(hist8.history['loss'], label = 'loss')
plt.plot(hist8.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()