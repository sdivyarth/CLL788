import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Q1 a.
data=pd.read_excel('data_1.xlsx')
# Scatter Plot y vs x    
plt.title('Q1 a. Scatter Plot y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(data.x,data.y)
plt.show()

#Histogram of x
plt.title('Q1 a. Histogram x')
plt.xlabel('x')
plt.ylabel('Number of occurances')
plt.hist(data.x)
plt.show()

#Histogram of y
plt.title('Q1 a. Histogram y')
plt.xlabel('y')
plt.ylabel('Number of occurances')
plt.hist(data.y)
plt.show()

#Heat map
hm=pd.DataFrame(np.corrcoef(data[['x','y']].T),columns=['x','y'])
hm.index=['x','y']
sns.heatmap(hm)


#Box plots
sns.boxplot(y=data.x).set_title('Box Plot of data_1 x')
plt.show()

sns.boxplot(y=data.y).set_title('Box Plot of data_1 y')
plt.show()



#Q1 b.
data=pd.read_excel('data_3.xlsx')
# Scatter Plot y vs x    

plt.title('Q1 b. Scatter Plot y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(data.x,data.y)
plt.show()

#Histogram of x
plt.title('Q1 b. Histogram x')
plt.xlabel('x')
plt.ylabel('Number of occurances')
plt.hist(data.x)
plt.show()

#Histogram of y
plt.title('Q1 b. Histogram y')
plt.xlabel('y')
plt.ylabel('Number of occurances')
plt.hist(data.y)
plt.show()

#Heat map
hm=pd.DataFrame(np.corrcoef(data[['x','y']].T),columns=['x','y'])
hm.index=['x','y']
sns.heatmap(hm)


#Box plots
sns.boxplot(y=data.x).set_title('Box Plot of data_3 x')
plt.show()

sns.boxplot(y=data.y).set_title('Box Plot of data_3 y')
plt.show()


#Q3
def get(x):
    print('Kurtoisis',end=' ')
    print(x.kurt())
    print('Standard deviation',end=' ')
    print(x.std())
    print('Skewness',end=' ')
    print(x.skew())
    print('Mean',end=' ')
    print(x.mean())
    print('Median',end=' ')
    print(x.median())

data=pd.read_excel('data_1.xlsx')
print('Q3.1 Statistics for data_1')
print('\nx',end='\n\n')
get(data['x'])
print('\ny',end='\n\n')
get(data['y'])
print()

data=pd.read_excel('data_3.xlsx')
print('Q3.2 Statistics for data_3')
print('\nx',end='\n\n')
get(data['x'])
print('\ny',end='\n\n')
get(data['y'])



#Q4
print('Outliers data_1 using Standard Deviation Method')
data=pd.read_excel('data_1.xlsx')
pd.concat([data[abs(data.x-data.x.mean())>3*data.x.std()],data[abs(data.y-data.y.mean())>3*data.y.std()]])
print(print('Outliers data_3 using Standard Deviation Method'))
data=pd.read_excel('data_3.xlsx')
pd.concat([data[abs(data.x-data.x.mean())>3*data.x.std()],data[abs(data.y-data.y.mean())>3*data.y.std()]])

print()
print('Outliers data_1 using MAD')
data=pd.read_excel('data_1.xlsx')
x_mad,y_mad=abs(data.x-data.x.median()).median(),abs(data.y-data.y.median()).median()
pd.concat([data[0.6745*(data.x-data.x.median())/x_mad>3.5],data[0.6745*(data.y-data.y.median())/y_mad>3.5]]).drop_duplicates()

print('Outliers data_3 using MAD')
data=pd.read_excel('data_3.xlsx')
x_mad,y_mad=abs(data.x-data.x.median()).median(),abs(data.y-data.y.median()).median()
pd.concat([data[0.6745*(data.x-data.x.median())/x_mad>3.5],data[0.6745*(data.y-data.y.median())/y_mad>3.5]]).drop_duplicates()
