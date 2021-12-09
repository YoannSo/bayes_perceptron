import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.linear_model import SGDClassifier
from keras.layers import Dense, Activation, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from numpy.random import seed
from tensorflow import set_random_seed
train = pd.read_csv('Entrainement.csv',delimiter=';') #delimiter=';' muyimportante!!!
#print(train.columns.values) #Affiche les colonnes de la data
print(train.describe())
#print(train.head())#voir la data
#print(train.tail())
print(train.info())
print(train.isnull().sum())#Aucune valeur non fournie dans la dataset..merci karim
train.rename(columns={"GOAL-Wine Quality": "GOAL_Wine_Quality"},inplace=True)

"""
print(train.head())#voir la data
sns.countplot(x='GOAL_Wine_Quality', data=train, palette='hls', hue='Alcohol
(%vol)')#GOAL_Wine_Quality
plt.xticks(rotation=45)
plt.show()
"""
print (train['GOAL_Wine_Quality'].value_counts())
#To show the histogram of quality variable
fig = px.histogram(train, x = 'GOAL_Wine_Quality')
fig.show()
"""
corr = train.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
"""
fig2 = plt.figure(figsize=(10,6))
sns.barplot(x= 'GOAL_Wine_Quality', y ='Fixed acidity (g/dm3)', data = train)
plt.show()
fig3 = plt.figure(figsize=(10,6))
sns.barplot(x= 'GOAL_Wine_Quality', y ='Alcohol (%vol)', data = train)
plt.show()
c = pd.Categorical(train['GOAL_Wine_Quality'])
print(c)
label_quality = LabelEncoder()
train['GOAL_Wine_Quality'] =label_quality.fit_transform(train['GOAL_Wine_Quality'])
print(train['GOAL_Wine_Quality'].value_counts())
fig4 = plt.figure(figsize=(10, 6))
sns.countplot(train['GOAL_Wine_Quality'])
plt.show()
#Now seperate the dataset as response variable and feature variabes
X = train.drop('GOAL_Wine_Quality', axis = 1)
y = train['GOAL_Wine_Quality']
#print(X)
#print(y)
#Train and Test splitting of data
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
#print ('trainShape',train.shape[1])
def neuralNetwork_model():
    model = Sequential()
    model.add(Dense(512, input_dim=11,
    kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    optimizer='adam', metrics=['accuracy'])
    return model

model = neuralNetwork_model()
print(model.summary())
training = model.fit(X_train, y_train, epochs=10, batch_size=32,validation_data=(X_test,y_test), verbose=1)
val_acc = np.mean(training.history['val_acc'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))