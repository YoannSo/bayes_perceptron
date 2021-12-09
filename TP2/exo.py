# Apprentissage des chiffres : Base MNIST – Version avec KERAS

import numpy as np
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
from tensorflow.keras import optimizers


from keras.datasets import mnist
from keras.utils import np_utils
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import plotly.express as px
import pandas as pd

data = pd.read_csv ('train.csv')  
""" Verification des valeurs
print(data.columns.values) #Affiche les colonnes de la data
print(data.describe())
print(data.head())#voir la data
print(data.tail())
print(data.info())
print(data.isnull().sum())
"""
data.rename(columns={"GOAL-Spam": "GOAL_Spam"},inplace=True)

X = data.drop('GOAL_Spam', axis = 1)
y = data['GOAL_Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#Let's see how our model performed
resultRFC = classification_report(y_test, pred_rfc)
print(resultRFC)
#Stochastic Gradient Decent Classifier¶
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
resultSGDC = classification_report(y_test, pred_sgd)
print(resultSGDC)

def neuralNetwork_model():
    model = Sequential()
    model.add(Dense(512, input_dim=11,kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

model = neuralNetwork_model()
print(model.summary())
training = model.fit(X_train, y_train, epochs=10, batch_size=32,validation_data=(X_test,y_test), verbose=1)
val_acc = np.mean(training.history['val_acc'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))