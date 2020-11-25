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
    
    def fit(self, x, y):
        b = 0
        m = 0
        n = x.shape[0]
        for _ in range(self.iterations):
            b_gradient = -2 * np.sum(y - m*x + b) / n
            m_gradient = -2 * np.sum(x*(y - (m*x + b))) / n
            b = b + (self.learning_rate * b_gradient)
            m = m - (self.learning_rate * m_gradient)
        self.theta1, self.theta0 = m, b

    def save_theta(self, x, y):
	    self.theta1 = (y[0] - y[1]) / (x[0] - x[1])
	    self.theta0 = y[0] - self.theta1 * x[0]
	    theta = [self.theta0, self.theta1]
	    np.savetxt("theta.csv", theta, delimiter = ',')
        
    def estimatePrice(self, x):
        return self.theta1 * x + self.theta0
    
def main():
    data = np.loadtxt("data.csv", dtype = np.longdouble, delimiter = ',', skiprows = 1)
    plt.scatter(data[:,0], data[:,1], color='black')

    x = standardize(data[:,0])
    y = standardize(data[:,1])

    clf = GradientDescentLinearRegression()
    clf.fit(x, y)
    plt.style.use('classic')
    y = clf.estimatePrice(x)

    x = destandardize(x, data[:,0])
    y = destandardize(y, data[:,1])

    clf.save_theta(x, y)
    # windows
    plt.plot(x, y)
    # mac
    # plt.axeline((x[0],y[0]),(x[1],y[1]))
    plt.gca().set_title("Gradient Descent Linear Regressor")
    # windows
    plt.savefig("matplotlib.png")

    # mac
    # plt.show()


if __name__ == '__main__':
    main()