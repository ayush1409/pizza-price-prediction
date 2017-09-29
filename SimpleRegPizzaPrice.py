import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation

# Training sets
X = [[6],[8],[10],[14],[18]]
y = [[7], [9], [13], [17.5], [18]]

plt.figure()

plt.title('Pizza price plotted against Price')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in $')

plt.plot(X,y,'k. ')

plt.axis([0,25,0,25])
plt.grid(True)
plt.show()

model = LinearRegression()

model.fit(X,y)

x = input('Enter the diameter of pizza(in inches) : ')
print 'A %.2f" pizza should cost: $%.2f' % (x, model.predict([[x]]))

print 'Cost function value : %.2f' % np.mean((model.predict(X)-y)**2)

#print 'Variance : %.2f' % np.var(X, ddof=1)

a = np.array(X).ravel()		# need to change the array from 2d to 1d to calculate covariance  

b = np.array(y).ravel()	

#print 'CoVariance : %.2f' % np.cov(a,b)[0][1]

# taking the regression model as y = alpha*x + beta
# using beta = covariance/variance

beta = np.cov(a,b)[0][1]/np.var(X,ddof = 1)

# as the model passes through mean(X) and mean(y)

alpha = (np.mean(y)-beta)/np.mean(X);

print 'the regression model is Price = %.2f * Diameter + ' % alpha,beta,' with minimum cost'
