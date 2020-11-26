import numpy as np
import sys
import os.path
from os import path
import array as arr

def estimate_price(theta_0, theta_1, x):
	return theta_0 + theta_1 * x

def main():
	
	if(path.exists('theta.csv')):
		theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')
	else:
		theta = arr.array('d',[0,0])
	while 1:
		print('Provide the mileage: ')
		try:
			mileage = input()
			mileage = int(mileage)
			if mileage >= 0:
				break
			else:
				print('Input should be a positive integer.')
		except:
			sys.exit('Error input.')
	print(estimate_price(theta[0], theta[1], mileage))

if __name__ == "__main__":
	main()
