import numpy as np
from numpy.lib.polynomial import polydiv
import tensorflow
import keras
import matplotlib.pyplot as plt
import random

def perceptron(x,w,active):
    temp = 0
    for i in range(0,len(x)):
        temp+= x[i]*w[i+1]
    temp+= w[0]
    if (active==0):
        return np.sign(temp),w
    else :
        return np.tanh(temp),w


def apprentissage(x,yd,active):
    w = {1,5,6} # Le probleme viens d'ici
    print(w[0,0])
    for i in range (0,len(x[0])):
        tempW = [0,0,0]
        point = [x[0][i],x[1][i]]
        result,poids=perceptron(point,tempW,active)
        if(result!=yd[i]):
            tempW[1]= poids[1] + (0.1*(yd[i]-result)*point[0])
            tempW[2] = poids[2] + (0.1*(yd[i]-result)*point[1])
            tempW[0] =  tempW[0] + 0.1*(yd[i]-result)
        w[0][0] = tempW[0]
        w[1][0] = tempW[1]
        w[2][0] = tempW[2]
    
    return w, 0
   
def affiche_classe(x,clas,K,w):
    t=[np.min(x[0,:]),np.max(x[0,:])]
    z=[(-w[0,0]-w[0,1]*np.min(x[0,:]))/w[0,2],(-w[0,0]-w[0,1]*np.max(x[0,:]))/w[0,2]]
   
    plt.plot(t,z);
    
    ind=(clas==-1)
    plt.plot(x[0,ind],x[1,ind],"o")
    
    ind=(clas==1)
    plt.plot(x[0,ind],x[1,ind],"o")
    
    plt.show()
# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]  # 
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]  # 
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128)-1,np.ones(128)))
w,mdiff=apprentissage(data,oracle,1)
plt.plot(mdiff)
plt.show()
affiche_classe(data,oracle,2,w)