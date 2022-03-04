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



model5 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape, kernel_regularizer='l2'),
    tf.keras.layers.Dense(512//2, activation='tanh'),
    tf.keras.layers.Dense(512//4, activation='tanh'),
    tf.keras.layers.Dense(512//8, activation='tanh'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model5.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])
hist5 = model5.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test), verbose=False)


loss5, acc5, mse5 = model5.evaluate(X_test, y_test)
print(f"Loss is {loss5},\nAccuracy is {acc5 * 100},\nMSE is {mse5}")




model6 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape, kernel_regularizer='l2'),
    tf.keras.layers.Dense(512//2, activation='tanh', kernel_regularizer='l2'),
    tf.keras.layers.Dense(512//4, activation='tanh', kernel_regularizer='l2'),
    tf.keras.layers.Dense(512//8, activation='tanh', kernel_regularizer='l2'),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model6.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])
hist6 = model6.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test), verbose=False)



loss6, acc6, mse6 = model6.evaluate(X_test, y_test)
print(f"Loss is {loss6},\nAccuracy is {acc6 * 100},\nMSE is {mse6}")





plt.figure(figsize=(15,8))
plt.plot(hist5.history['loss'], label = 'loss')
plt.plot(hist5.history['val_loss'], label='val loss')
plt.title("hist5 Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(15,8))
plt.plot(hist5.history['acc'], label = 'ACC')
plt.plot(hist5.history['val_acc'], label='val acc')
plt.title("hist5 acc vs Val_acc")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.show()




plt.figure(figsize=(15,8))
plt.plot(hist6.history['loss'], label = 'loss')
plt.plot(hist6.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(15,8))
plt.plot(hist6.history['acc'], label = 'ACC')
plt.plot(hist6.history['val_acc'], label='val acc')
plt.title("acc vs Val_acc")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.show()