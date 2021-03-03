import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split

def plot(x,y,x_name='',y_name='',title=''):
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
    
data=pd.read_excel('q2train.xlsx')
data=data.values
y_train=data[:,-1]
X_train=np.concatenate((np.ones((len(y_train),1)),data[:,:-1]),axis=1)


def sigmoid(z):
    return 1./(1.+np.exp(-z))

def loss(y_hat,y):
    return -(sum(np.multiply(y_train,np.log(y_hat)))+sum(np.multiply(1-y_train,np.log(1-y_hat))))

def batchGrad(X,y,w0,eta,iter0=100000,tol=1e-8):     
               
    w=w0
    history_loss=[]
    m=len(X)
    for _ in range(iter0):
        w0=w
        y_pred=sigmoid(X.dot(w))
        der=X.T.dot(y_pred-y)
        w=w-eta*der
        if sum(abs(w-w0))<tol:
            break
        if _==iter0-1:
            print('Weights did not converge after '+str(iter0)+' iterations')
        history_loss.append(loss(y_pred,y))
               
    plt.title('Loss vs iterations')
    plt.scatter([i for i in range(len(history_loss))],history_loss)
    return w    

def predict(X,w): 
    return np.array([1 if x>0.5 else 0 for x in X.dot(w)])
    

print('Batch Gradient Descent')
data=pd.read_excel('q2train.xlsx')
data=data.values
y_train=data[:,-1]
X_train=np.concatenate((np.ones((len(y_train),1)),data[:,:-1]),axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
w0=np.zeros((len(X_train[0]),))
w=batchGrad(X_train,y_train,w0,0.000005,1000000,2e-6)
y_pred=predict(X_train,w)
print('macro F1 score is',end=' ')
print(f1_score(y_train,y_pred, average='micro'))
print(f1_score(y_test,predict(X_test,w), average='micro'))
print(w)
