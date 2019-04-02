import numpy as nu
import math as mt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x=nu.linspace(0,1,100)
n=nu.random.normal(0,0.3,100)
s=b=z=y=nu.zeros(100)
for i in range(100):
	z[i]=2*mt.pi*x[i]
for i in range(100):
	y[i]=n[i]+mt.sin(z[i])
a=y
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, train_size=0.80)
plt.plot(x_train, y_train, 'o')
alpha=0.05
c =  nu.array([[0,0]]).T # Initial values of c
X = nu.c_[nu.ones(80),x_train]
Y = nu.c_[y_train]
W=[]
for i in range(9):
    X_i=nu.c_[x_train].T
for i in range(9):
    b[i]=nu.sum(c[1] - alpha * (1/len(Y)) * nu.sum(nu.dot(X_i,(nu.dot(X,c)-Y))))
    c= nu.array([[0],[b[i]]])
    W.append(b[i])
    s[i]=s[i]+b[i]
for i in range(9):
    plt.plot(x, a)
    a=a+pow(x,i)*s[i]
plt.suptitle('Linear regression by gradient descent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print("Estimated Value of W:\n",W)
t=nu.dot((nu.dot(X,c) - y_test).T,(nu.dot(X,c) - y_test))/(2*len(Y))
print("test_error_on_test_data =\n",t)