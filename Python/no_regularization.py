from tabnanny import verbose
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from load_files import load_files
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

# print(X)
# print(y)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape),
    tf.keras.layers.Dense(512//2, activation='tanh'),
    tf.keras.layers.Dense(512//4, activation='tanh'),
    tf.keras.layers.Dense(512//8, activation='tanh'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

print("")
print("Compiling")
# print(model1.summary())
model1.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])

print("")
print("Fitting")
hist = model1.fit(X_train, y_train, epochs=500, batch_size=20000, validation_data=(X_test,y_test),verbose=True)

print("")
print("Evaluating")
loss1, acc1, mse1 = model1.evaluate(X_test, y_test)

print("")
print(f"Loss is {loss1},\nAccuracy is {acc1*100},\nMSE is {mse1}")


plt.style.use('ggplot')
plt.plot(hist.history['loss'], label = 'loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(hist.history['acc'], label = 'acc')
plt.plot(hist.history['val_acc'], label='val acc')
plt.title("acc vs Val_acc")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.show()


