import pandas as pd
dataset = pd.read_csv('cancer.csv')
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =  train_test_split(x,y, test_size=0.2)

import tensorflow as tf
model = tf.keras.models.Sequential()

print("-------Creación capas de red neuronal")
model.add(tf.keras.layers.Dense(100, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print("-------Compilacion")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("-------Alimentar modelo")
model.fit(x_train, y_train, epochs=500)

print("-------Evaluar datos")
model.evaluate(x_test, y_test)