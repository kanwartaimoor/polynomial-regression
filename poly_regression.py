# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:09:20 2020

@author: Rai Kanwar Taimoor
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


df = np.loadtxt(fname = 'polynomialRegressionData.csv', delimiter =',', skiprows = 1,usecols=(0,1,))
Y = np.loadtxt(fname = 'polynomialRegressionData.csv', delimiter =',', skiprows = 1,usecols=(2))
xs=df[:,0]
xs.reshape(-1,1)
xs2=df[:,1]
xs2.reshape(-1,1)

#arranging data and plotting it
q= np.zeros(145).reshape(-1,1)
for iteration in range(145):
    q[iteration]=xs[iteration]*xs2[iteration]
Y.reshape(-1,1)
x=df
plt.plot(q,Y,'p')

poly=PolynomialFeatures(degree=4)
x=poly.fit_transform(q)
k=x


x_transpose = np.transpose(x)
x_transpose_dot_x = x_transpose.dot(x)
temp_1 = np.linalg.inv(x_transpose_dot_x)
temp_2=x_transpose.dot(Y)
theta =temp_1.dot(temp_2)
print("theeta values = ",theta,'\n')


Y_pred = (np.dot(x,theta).reshape(-1,1))

n = float(len(df))
cost = sum([data**2 for data in (Y-Y_pred)]) / n 


c= np.zeros(145)
for iteration in range(145):
    c[iteration]=xs[iteration]*xs2[iteration]   
x=c
y =Y



coefficients = np.polyfit(x, y, 4)
poly = np.poly1d(coefficients)
new_x = np.linspace(x[0], x[144])
new_y = poly(new_x)
plt.plot(x, y, "o", new_x, new_y)
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend() 
ax.scatter(xs, xs2, Y, c='r',label ='y',s = 50, marker='o')
ax.scatter(xs, xs2, Y_pred, c='b',label ='y_pred',s = 50, marker='o')



plt.show()












