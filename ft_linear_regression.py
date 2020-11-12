import numpy as np
# mac
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
# windows
import matplotlib
matplotlib.use('Agg')

def standardize(x):
	return (x - np.mean(x)) / np.std(x)

def destandardize(x, x_ref):
	return x * np.std(x_ref) + np.mean(x_ref)

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations
    
    def fit(self, X, y):
        b = 0
        m = 0
        n = X.shape[0]
        for _ in range(self.iterations):
            b_gradient = -2 * np.sum(y - m*X + b) / n
            m_gradient = -2 * np.sum(X*(y - (m*X + b))) / n
            b = b + (self.learning_rate * b_gradient)
            m = m - (self.learning_rate * m_gradient)
        self.theta1, self.theta0 = m, b
        
    def estimatePrice(self, X):
        return self.theta1 * X + self.theta0

if __name__ == '__main__':
    
    data = genfromtxt('data.csv',delimiter=',')
    data = np.delete(data, 0, 0)

    plt.scatter(data[:,1], data[:,0], color='black')

    X = standardize(data[:,1])
    y = standardize(data[:,0])

    clf = GradientDescentLinearRegression()
    clf.fit(X, y)

    plt.style.use('classic')

    y = clf.estimatePrice(X)

    X = destandardize(X, data[:,1])
    y = destandardize(y, data[:,0])
    
    plt.plot(X, y)
    plt.gca().set_title("Gradient Descent Linear Regressor")
    # windows
    plt.savefig("matplotlib.png")

    # mac
    # plt.show()
