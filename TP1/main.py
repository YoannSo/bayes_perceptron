from numpy.core.defchararray import array
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def plusProche(x,y,data,oracle):
    plusPres=[0,0,0]
    for i in range (1,len(data[0])):
        distance = ((x-data[0][i])**2+(y-data[1][i])**2)**0.5
        
        distanceTo0= ((x-data[0][plusPres[0]])**2+(y-data[1][plusPres[0]])**2)**0.5
        distanceTo1= ((x-data[0][plusPres[1]])**2+(y-data[1][plusPres[1]])**2)**0.5
        distanceTo2= ((x-data[0][plusPres[2]])**2+(y-data[1][plusPres[2]])**2)**0.5
       
        if(distance<distanceTo0):
            plusPres[2]=plusPres[1]
            plusPres[1]=plusPres[0]
            plusPres[0]=i
        elif (distance<distanceTo1):
            plusPres[2]=plusPres[1]
            plusPres[1]=i
        elif (distance<distanceTo2):
            plusPres[2]=i
    classe = 0
    classe+= oracle[plusPres[0]]
    classe+= oracle[plusPres[1]]
    classe+= oracle[plusPres[2]]
    if (classe<2): return 0
    else : return 1


def kppv(x,appren,oracle,K):

    clas = np.array([0]*len(x[0]))
    temp = 0
    for i in range (0,len(x[0])):
        temp = plusProche(x[0][i],x[1][i],appren,oracle)
        clas[i]=temp
    return clas
   
def affiche_classe(x,clas,K):
    for k in range(0,K):
        ind=(clas==k)
        plt.plot(x[0,ind],x[1,ind],"o")
    plt.show()


# Donnees de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]  
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))

mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]   
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))

data = np.concatenate((data1, data2), axis=1)
oracle = np.concatenate((np.zeros(128),np.ones(128)))

test1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 64))
test2 = np.transpose(np.random.multivariate_normal(mean2, cov2,64))

test = np.concatenate((test1,test2), axis=1)
K=3
clas=kppv(test,data,oracle,K)
affiche_classe(test,clas,2)