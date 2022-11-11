# https://github.com/tensorchiefs/dl_book/blob/master/chapter_05/nb_ch05_01.ipynb

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp
import evidential_deep_learning as edl
import shap

plt.style.use('default')

tfd = tfp.distributions
tfb = tfp.bijectors
print("TFP Version", tfp.__version__)
print("TF  Version", tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# custom functions

# Custom loss function to handle the custom regularizer coefficient
def EvidentialRegressionLoss(true, pred):
    return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)


# PREPARE DATA
wine_data = pd.read_csv('winequality-red.csv', parse_dates=True, encoding="cp1252")
print(wine_data.head())

feature_labels = ['fixed acidity',
                  'volatile acidity',
                  'citric acid',
                  'residual sugar',
                  'chlorides',
                  'free sulfur dioxide',
                  'total sulfur dioxide',
                  'density',
                  'pH',
                  'sulphates',
                  'alcohol']

X = wine_data[feature_labels]

y = wine_data['quality']

X = np.array(X).astype('float64')
y = np.array(y).astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create Model
n_units = 50
inputs = Input(shape=(11,))
x = BatchNormalization(axis=-1)(inputs)
for i in range(3):
    x = Dense(n_units, activation="selu")(x)
    #x=Dropout(0.5)(x)

out = edl.layers.DenseNormalGamma(1)(x)

model = Model(inputs=inputs, outputs=out)

model.compile(optimizer=Adam(learning_rate=1e-2),
              loss=EvidentialRegressionLoss)

model.summary()

my_callbacks = [
    EarlyStopping(patience=10, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
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

# Predict and plot using the trained model
y_pred = model(X_test)
#print(y_pred)

mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
mu = mu[:, 0]
var = np.sqrt(beta / (v * (alpha - 1)))
#print(f'mu={mu}')

sigma = np.sqrt(beta / (alpha - 1))
#print(f'sigma={sigma}')
std_of_mu = np.sqrt(beta / (v * (alpha - 1)))
#print(f'std_of_mu={std_of_mu}')
unc = np.sqrt(np.power(sigma, 2) + np.power(std_of_mu, 2))

# STACK RESULTS
#print(f'X_test={X_test}')
result = np.hstack((X_test,
                    np.array([mu]).T,
                    sigma,
                    std_of_mu,
                    unc))
#print(f'result={result}')
labels = feature_labels + ['prediction', 'aleatoric unc', 'epistemic unc', 'total unc']
#print(labels, len(labels))
df = pd.DataFrame(result, columns=labels)

#print(df)
print(df.head())
print(df.describe())
df.to_excel("output.xlsx")

