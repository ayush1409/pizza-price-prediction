
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

# Trianing data
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
# Testing data
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

model = LinearRegression()

model.fit(X_train,y_train)

xx = np.linspace(0, 26, 100)
yy = model.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx,yy)

#plt.show()

quad_featuriser = PolynomialFeatures(degree = 2)
X_train_quad = quad_featuriser.fit_transform(X_train)
X_test_quad = quad_featuriser.transform(X_test)

reg_quad = LinearRegression()
reg_quad.fit(X_train_quad, y_train)
xx_quad = quad_featuriser.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, reg_quad.predict(xx_quad), c='r', linestyle='--')

plt.title('Pizza price Regressed on Diameter')
plt.xlabel('Diameter in Inches')
plt.ylabel('Price in Dollars')
plt.axis([0,25,0,25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print X_train
print X_train_quad
print X_test
print X_test_quad

print 'Simple Linear Regression R sqaured : ', model.score(X_test,y_test)
print 'Quad Regression R squared : ', reg_quad.score(X_test_quad,y_test)
