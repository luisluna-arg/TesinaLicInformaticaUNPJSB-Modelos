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

# print("")
# print("model2 definition")
# model2 = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape, kernel_regularizer='l1'), #Only change is here where we add kernel_regularizer
#     tf.keras.layers.Dense(512//2, activation='tanh'),
#     tf.keras.layers.Dense(512//4, activation='tanh'),
#     tf.keras.layers.Dense(512//8, activation='tanh'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(5, activation='softmax')
# ])

# print("")
# print("model2 compile")
# model2.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])

# print("")
# print("model2 fit")
# hist2 = model2.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test))

# loss2, acc2, mse2 = model2.evaluate(X_test, y_test)
# print("")
# print(f"Loss is {loss2},\nAccuracy is {acc2 * 100},\nMSE is {mse2}")


# plt.plot(hist2.history["loss"], label = "loss")
# plt.plot(hist2.history["val_loss"], label="val loss")
# plt.title("Loss vs Val_Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# plt.figure(figsize=(15,8))
# plt.plot(hist2.history['acc'], label = 'acc')
# plt.plot(hist2.history['val_acc'], label='val acc')
# plt.title("acc vs Val_acc")
# plt.xlabel("Epochs")
# plt.ylabel("acc")
# plt.legend()
# plt.show()





print("")
print("model3 definition")
model3 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='tanh', input_shape = X_train[0].shape, kernel_regularizer='l1'),
    tf.keras.layers.Dense(512//2, activation='tanh', kernel_regularizer='l1'),
    tf.keras.layers.Dense(512//4, activation='tanh', kernel_regularizer='l1'),
    tf.keras.layers.Dense(512//8, activation='tanh', kernel_regularizer='l1'),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l1'),
    tf.keras.layers.Dense(5, activation='softmax')
])

print("")
print("model3 compile")
model3.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse'])

print("")
print("model3 fit")
hist3 = model3.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test), verbose=2)

loss3, acc3, mse3 = model3.evaluate(X_test, y_test)
print("")
print(f"Loss is {loss3},\nAccuracy is {acc3 * 100},\nMSE is {mse3}")


plt.figure(figsize=(15,8))
plt.plot(hist3.history['loss'], label = 'loss')
plt.plot(hist3.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(15,8))
plt.plot(hist3.history['acc'], label = 'ACC')
plt.plot(hist3.history['val_acc'], label='val acc')
plt.title("acc vs Val_acc")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.show()


