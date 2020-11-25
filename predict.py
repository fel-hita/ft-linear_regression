import numpy as np

def estimate_price(theta_0, theta_1, x):
	return theta_0 + theta_1 * x

def main():
	theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')
	while 1:
		print('Provide the mileage: ')
		try:
			mileage = input()
		except:
			sys.exit('Error on input. Exit...')
		try:
			mileage = int(mileage)
			if mileage >= 0:
				break
			else:
				print('Invalid input.')
		except ValueError:
			print('Invalid input.')

	print(estimate_price(theta[0], theta[1], mileage))

if __name__ == "__main__":
	main()
