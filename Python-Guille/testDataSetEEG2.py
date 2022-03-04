import matplotlib.pyplot as plt
import numpy as np 
# import os
# import pandas as pd
import tensorflow as tf
# # import tensorflow.keras.losses as loss
# # import tensorflow.keras.metrics as metrics
# # import tensorflow.keras.optimizers as optimizers
# import tensorflow as tf
# import sklearn
# import seaborn as sns

# # import os.path
# # if not os.path.exists('mlp_helper.py'):
# #     !wget https://github.com/lab-ml-itba/MLP-2019/raw/master/mlp_helper.py

# from mlp_helper import plot_boundaries_keras, get_dataset, plot_boundaries, draw_neural_net, return_weights_notation, get_dataset_2

# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization
# # from tensorflow.keras.optimizers import Adam, SGD
# from keras.utils.np_utils import to_categorical

# # from tensorflow.keras.utils import Sequence
# class SeqGen(tf.keras.utils.Sequence):

#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size
        
#     def __len__(self):
#         return int(np.ceil(len(self.x) / float(self.batch_size)))
    
#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return batch_x, batch_y

# def get_model():
#     model = Sequential()
#     model.add(LSTM(16,input_shape=(11,11), return_sequences=True))    
#     model.add(LSTM(16))  
#     model.add(BatchNormalization())  
#     model.add(Dense(4, activation = None))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.compile(loss=loss.CategoricalCrossentropy(from_logits=True),optimizer=optimizers.Adam(), metrics=[metrics.SparseCategoricalAccuracy()])
#     model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#     return model

# def plot_history_accuracy(history):
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()

# def plot_history_loss(history):
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()

# def train_model(model, X, Y):
#     x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42, shuffle=True )
#     standard = StandardScaler().fit(x_train)
#     x_train_standard = standard.transform(x_train).reshape(x_train.shape[0],)#11,11)
#     x_test_standard = standard.transform(x_test).reshape(x_test.shape[0],)#11,11)
#     print(x_train_standard)
#     train_labels = np.argmax(x_train, axis=1)

#     return model.fit(SeqGen(x_train_standard,y_train,batch_size=12), validation_data=(x_test_standard,y_test), epochs=50, verbose=1)


dataset = pd.read_csv('arriba_abajo_derecha_izquierda5.csv', delimiter=',')


df = pd.read_csv('arriba_abajo_derecha_izquierda3.csv', delimiter=',', index_col=False)
df.dataframeName = 'dataset.csv'


# target = 'class'


# col = dataset.columns
# features = col[1:]
# print(features)


# dataset[target].value_counts()


# sns.countplot(x=target, data=dataset, palette="bone")
# plt.show()



# #plotScatterMatrix(dataset, 20, 10)


# dataset.describe()



# list_cor = pd.DataFrame(dataset[features].corr().unstack().abs().sort_values().drop_duplicates())
# list_cor.columns = ['correlation_index']
# list_corr_high = list(list_cor[-33:-1]['correlation_index'].index)
# list_corr_high




# total = dataset[features].isnull().sum().sort_values(ascending = False)
# percent = (dataset[features].isnull().sum()/dataset[features].isnull().count()*100).sort_values(ascending = False)
# missing  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# missing



# def preprocess_inputs(df, features, target):        
#     list_cor = pd.DataFrame(df[features].corr().unstack().abs().sort_values().drop_duplicates())
#     list_cor.columns = ['correlation_index']
#     list_corr_high = list(list_cor[-33:-1]['correlation_index'].index)
#     list_corr_high
#     for eletrods in list_corr_high:
#         df['__'.join(list(eletrods))] = df.apply(lambda row: abs(row[eletrods[0]] - row[eletrods[1]]), axis=1)
    
#     col = df.columns       # .columns gives columns names in data
#     features = col[1:]
#     print(features)
#     y = df.drop(features, axis=1)
#     y = to_categorical(y)
#     X = df[features]
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
#     # Scale X with a standard scaler
#     transformer = StandardScaler() 

#     X_train_transformer = transformer.fit_transform(X_train)
#     X_test_transformer = transformer.transform(X_test)

#     return X_train_transformer, X_test_transformer, y_train, y_test



# X_train, X_test, y_train, y_test = preprocess_inputs(dataset, features, target)



# X_train.shape




# y_train.shape




# def build_model(X):  
#     k2 = int(X.shape[1]**(1/2))
#     #print(k2)
#     inputs = tf.keras.Input(shape=(X.shape[1]),name="inputs")
#     tensorReshape = tf.reshape(X, (-1,k2, k2), name="expand_dims")
#     print(tensorReshape.shape)
#     expand_dims = tf.reshape(inputs, (-1,k2, k2), name="expand_dims")
#     #print(expand_dims.shape)
#     lstm = tf.keras.layers.LSTM(4, return_sequences=True)(expand_dims)
#     drop = tf.keras.layers.Dropout(.5)(lstm)
#     print(drop)
#     lstm = tf.keras.layers.LSTM(4, return_sequences=True)(expand_dims)
#     drop = tf.keras.layers.Dropout(.5)(lstm)
#     print(drop)
#     flatten = tf.keras.layers.Flatten()(lstm)
#     flatten.shape
#     outputs = tf.keras.layers.Dense(4, activation='softmax')(flatten)  
#     #print(outputs)  
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     model.compile(
#         optimizer='sgd',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model





# X_train.shape




# y_train.shape




# class_model = build_model(X_train)





# def train_model(dataset, features, target, build_model=build_model):
#     X_train, X_test, y_train, y_test = preprocess_inputs(dataset, features, target)

#     class_model = build_model(X_train)

#     history = class_model.fit(
#         X_train,
#         y_train,
#         validation_split=0.3,
#         batch_size=32,
#         epochs=30,
#         verbose=0,
#         callbacks=[
#             tf.keras.callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=3,
#                 restore_best_weights=True
#             )
#         ]
#     )

#     print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(history.history['accuracy'])*100, np.std(history.history['accuracy'])*100)) 

#     class_acc = class_model.evaluate(X_test, y_test, verbose=0)[1]
#     print("Test Accuracy (Class Model): {:.2f}%".format(class_acc * 100))
    
#     y_pred = np.array(list(map(lambda x: np.argmax(x), class_model.predict(X_test))))
#     clr = sklearn.metris.classification_report(y_test.argmax(axis=-1) , y_pred)
# #     print("Classification Report:\n----------------------\n", clr)
    
#     return history



# class_model = build_model(X_train)
# class_model.summary()




# X_train.shape
# y_train.shape
# print(y_train)




# class_model.fit(
#         X_train,
#         y_train,
#         validation_split=0.3,
#         batch_size=29,
#         epochs=25,
#         verbose=1,
#         callbacks=[
#             tf.keras.callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=3,
#                 restore_best_weights=True
#             )
#         ]
#     )



# history = train_model(dataset, features, target)
# # plot_accuracy_history(history)
# # plot_loss_history(history)



