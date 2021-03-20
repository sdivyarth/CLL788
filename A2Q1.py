import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

data=pd.read_excel('q1train.xlsx')
t=data.Label.apply(lambda x: 1 if x==0 else -1).values #Class 1: 0->1 ;;; Class2 1->-1
X=data[data.columns[:-1]].values
X=np.concatenate((np.ones((len(t),1)),data.values[:,:-1]),axis=1)

def plot(x,y,x_name='',y_name='',title='',ss='q2q1.png'):
    plt.scatter(x,y,s=1)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(ss)
    plt.show()
    
def visualise(w,X,t,name='q1a2.png'):    
    x1=X[:,1]
    x2=X[:,2]
    plt.scatter(x1[t==1],x2[t==1],color='blue')
    plt.scatter(x1[t==-1],x2[t==-1],color='red')
    plt.legend(['Positive: 1','Negative: -1'])
    plt.title('Final Decision Boundary')
    plt.xlabel('Score 1: Aptitude')
    plt.ylabel('Score 2: Verbal')
    Y=-(w[0]/w[2]+w[1]*x1/w[2])
    plt.plot(x1,Y)
    plt.xlim([min(x1),max(x1)])
    plt.ylim([min(x2),max(x2)])
    plt.savefig(name)
    
def predict(X,w):
    return f(X.dot(w))

def f(x):
    return np.array(list(map(lambda y: 1 if y>=0 else -1,x)))

def perceptron(X,t,eta=0.000005,iter0=100000,tol=1e-6):
    
    w=np.array([1.0]*len(X[0])).reshape((len(X[0]),1))    
    cost=[]
    
    min_cost=float('inf')
    opt_w=None
    
    for _ in range(iter0):
        y=X.dot(w).flatten()
        M=(np.multiply(t,f(y))==-1) #Set of incorrectly classified points
        if len(M)==0:
            break
        cost.append(-sum(np.multiply(t[M],y[M])))
        if cost[-1]<min_cost:
            min_cost=cost[-1]
            opt_w=w
        index=_%sum(M)
        der=(eta*X[M][index]*t[M][index]).reshape(w.shape)
        if np.all(abs(der)<tol):
            break        
        w+=der
    plot(np.arange(iter0),np.array(cost),'Iteration','Perceptron Cost')
    
    return opt_w


w=perceptron(X,t,0.00005,200000,1e-5)
y=predict(X,w)
print(f1_score(t,y))
visualise(w,X,t)

#Test Data

data=pd.read_excel('q1test.xlsx')
X_test=np.concatenate((np.ones((len(data),1)),data),axis=1)
model_pred=predict(X_test,w)
visualise(w,X_test,model_pred)
np.savetxt('output1.txt',model_pred,fmt ='%.0f')
