# Apprentissage des chiffres : Base MNIST – Version avec KERAS

import numpy as np
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from tensorflow.keras import optimizers

from keras.datasets import mnist
from keras.utils import np_utils
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import plotly.express as px
import pandas as pd

data = pd.read_csv ('train.csv')

data.rename(columns={"GOAL-Spam": "GOAL_Spam"},inplace=True)

X = data.drop('GOAL_Spam', axis = 1)
y = data['GOAL_Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model_Gaussian = GaussianNB()
#training
model_Gaussian.fit(X_train, y_train)
#prédiction
prediction = model_Gaussian.predict(X_test)
#evaluation du modèle
precision = accuracy_score(y_test, prediction)*100
print(precision)

