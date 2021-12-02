import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import random
import os
def kppv(x,appren,oracle,K):
       return 0
   
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
data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128),np.ones(128)))
test1=np.transpose(np.random.multivariate_normal(mean1, cov1, 64))
test2=np.transpose(np.random.multivariate_normal(mean2, cov2,64))
test=np.concatenate((test1,test2), axis=1)
K=3
clas=kppv(test,data,oracle,K)
affiche_classe(test,clas,2)