# https://github.com/tensorchiefs/dl_book/blob/master/chapter_05/nb_ch05_01.ipynb

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp

plt.style.use('default')

tfd = tfp.distributions
tfb = tfp.bijectors
print("TFP Version", tfp.__version__)
print("TF  Version", tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# custom functions
from my_tools import plot_history_loss_accuracy

# Listing 5.4:Using a NN with a hidden layer for linear regression with non-constant variance

# PREPARE DATA
wine_data = pd.read_csv('winequality-red.csv', parse_dates=True, encoding="cp1252")
print(wine_data.head())

X = wine_data[['fixed acidity',
               'volatile acidity',
               'citric acid',
               'residual sugar',
               'chlorides',
               'free sulfur dioxide',
               'total sulfur dioxide',
               'density',
               'pH',
               'sulphates',
               'alcohol']]

y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create Model
n_units = 50
inputs = Input(shape=(11,))
x = BatchNormalization(axis=-1)(inputs)
x = Dense(n_units, activation="selu")(x)
x = Dense(n_units, activation="selu")(x)
x = Dense(n_units, activation="selu")(x)
out = Dense(1)(x)
model = Model(inputs=inputs, outputs=out)

model.compile(optimizer=Adam(learning_rate=1e-2),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.summary()

my_callbacks = [
    EarlyStopping(patience=10,monitor='val_mean_absolute_error'),
    ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.2, patience=5, min_lr=0.0001)
]

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=5000,
    callbacks=my_callbacks,
    validation_split=0.2,
    # validation_data=None,
    shuffle=True,
)

# Plot training
plot_history_loss_accuracy(history, acc='mean_absolute_error', val_acc='val_mean_absolute_error', first_sample=10)
print('\n Evaluating')
model.evaluate(X_test,y_test)

