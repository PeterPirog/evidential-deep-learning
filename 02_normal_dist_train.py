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
from tensorflow.keras.layers import Dense, BatchNormalization,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# custom functions
from my_tools import plot_history_loss_accuracy

def NLL(y, distr):
    return -distr.log_prob(y)

def my_dist(params):
    #return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable
    return tfd.Normal(loc=params[:,0:1], scale= params[:,1:2])# both parameters are learnable

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

X=np.array(X).astype('float64')
y=np.array(y).astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create Model
n_units = 50
inputs = Input(shape=(11,))
x = BatchNormalization(axis=-1)(inputs)
x = Dense(n_units, activation="selu")(x)
x = Dense(n_units, activation="selu")(x)
x = Dense(n_units, activation="selu")(x)
out_mean = Dense(1)(x)
out_std = Dense(1,activation='softplus')(x)

params = Concatenate()([out_mean,out_std]) #C
dist = tfp.layers.DistributionLambda(my_dist)(params)
print(f'dist type:{dir(dist)}')

model = Model(inputs=inputs, outputs=dist)
model_param= Model(inputs=inputs, outputs=params)

model.compile(optimizer=Adam(learning_rate=1e-2),
              loss=NLL,
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

#model_mean = Model(inputs=inputs, outputs=dist.mean())
#model_std = Model(inputs=inputs, outputs=dist.stddev())

# Plot training
plot_history_loss_accuracy(history, acc='mean_absolute_error', val_acc='val_mean_absolute_error', first_sample=10)
print('\n Evaluating')
model.evaluate(X_test,y_test)
output=model_param.predict(X_train)
print(output)

"""
# ds = tfds.load('wine_quality', split='train', shuffle_files=True)
(ds_train, ds_test) = tfds.load('wine_quality',
                                split=['train', 'test'],
                                shuffle_files=True,
                                as_supervised=True)

# Build your input pipeline
ds = ds_train.shuffle(1024).batch(1).prefetch(tf.data.AUTOTUNE)
for example in ds.take(1):
    features, quality = example["features"], example["quality"]
    print(features)
    print(quality)

# Define model



def NLL(y, distr):
  return -distr.log_prob(y)

def my_dist(params):
  return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable

inputs = Input(shape=(1,))
out1 = Dense(1)(inputs) #A
hidden1 = Dense(30,activation="relu")(inputs)
hidden1 = Dense(20,activation="relu")(hidden1)
hidden2 = Dense(20,activation="relu")(hidden1)
out2 = Dense(1)(hidden2) #B
params = Concatenate()([out1,out2]) #C
dist = tfp.layers.DistributionLambda(my_dist)(params)

model_flex_sd = Model(inputs=inputs, outputs=dist)
model_flex_sd.compile(Adam(learning_rate=0.01), loss=NLL)
#A The first output models the mean, no hidden layers are used
#B The second output models the spread of the distribution. Three hidden layers are used for it
#C Combining the outputs for the mean and the spread

model_flex_sd.summary()

history = model_flex_sd.fit(x_train, y_train, epochs=2000, verbose=0, validation_data=(x_val,y_val))

model_flex_sd_mean = Model(inputs=inputs, outputs=dist.mean())
model_flex_sd_sd = Model(inputs=inputs, outputs=dist.stddev())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('NLL')
plt.xlabel('Epochs')
plt.ylim(0,10)
plt.show()

# Result: flexible sigma

print(model_flex_sd.evaluate(x_train,y_train, verbose=0))
print(model_flex_sd.evaluate(x_val,y_val, verbose=0))

plt.figure(figsize=(14,5))

x_pred = np.arange(-1,6,0.1)

plt.subplot(1,2,1)
plt.scatter(x_train,y_train,color="steelblue") #observerd
preds = model_flex_sd_mean.predict(x_pred)
plt.plot(x_pred,preds,color="black",linewidth=2)
plt.plot(x_pred,preds+2*model_flex_sd_sd.predict(x_pred),color="red",linestyle="--",linewidth=2)
plt.plot(x_pred,preds-2*model_flex_sd_sd.predict(x_pred),color="red",linestyle="--",linewidth=2)
plt.xlabel("x",size=16)
plt.ylabel("y",size=16)
plt.title("train data")
plt.xlim([-1.5,6.5])
plt.ylim([-30,55])

plt.subplot(1,2,2)
plt.scatter(x_val,y_val,color="steelblue") #observerd
plt.plot(x_pred,preds,color="black",linewidth=2)
plt.plot(x_pred,preds+2*model_flex_sd_sd.predict(x_pred),color="red",linestyle="--",linewidth=2)
plt.plot(x_pred,preds-2*model_flex_sd_sd.predict(x_pred),color="red",linestyle="--",linewidth=2)
plt.xlabel("x",size=16)
plt.ylabel("y",size=16)
plt.title("validation data")
plt.xlim([-1.5,6.5])
plt.ylim([-30,55])
plt.show()

"""
