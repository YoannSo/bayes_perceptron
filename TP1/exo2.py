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


def apprentissage(x,yd,active,dataX,oracleX):
    w = np.array([[0,0,0]]) # Le probleme viens d'ici
    tempW = [50,150,15]
    err = []
    errX = []
    
    for j in range (0,10):
        y=[]
        yX = []
        error=0
        errorX=0
        for i in range (0,len(x[0])):
            point = [x[0][i],x[1][i]]
            pointX =[dataX[0][i],dataX[1][i]]
            result,poids=perceptron(point,tempW,active)
            resultX,poidsX=perceptron(pointX,tempW,active)
            y.append(result);
            yX.append(resultX);
            
            if(result!=yd[i]):
                
                tempW[1]= poids[1] + (0.1*(yd[i]-result)*point[0])
                tempW[2]= poids[2] + (0.1*(yd[i]-result)*point[1])
                tempW[0]=  tempW[0] + 0.1*(yd[i]-result)
        
        for z in range (0,len(y)):
            error+=pow((yd[z]-y[z]),2)
            errorX+=pow((oracleX[z]-yX[z]),2)
        err.append(error)
        errX.append(errorX)
        w[0,0] = tempW[0]
        w[0,1] = tempW[1]
        w[0,2] = tempW[2]
    print(len(err),len(x[0]))
    return w, err,errX
   
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
cov1 = [[4, 0], [0, 4]]  # 
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]  # 
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))

data3 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
data4 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
dataX=np.concatenate((data3, data4), axis=1)
oracleX=np.concatenate((np.zeros(128)-1,np.ones(128)))

data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128)-1,np.ones(128)))
w,mdiff,errX=apprentissage(data,oracle,0,dataX,oracleX)
plt.plot(mdiff)
plt.plot(errX)
plt.show()


affiche_classe(data,oracle,2,w)