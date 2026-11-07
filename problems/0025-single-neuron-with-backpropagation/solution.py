import numpy as np

def train_neuron(
	features: np.ndarray, 
	labels: np.ndarray, 
	initial_weights: np.ndarray, 
	initial_bias: float, 
	learning_rate: float, 
	epochs: int) -> (np.ndarray, float, list[float]):

	w, b = initial_weights, initial_bias
	n = len(features)
	mse_values = []

	for _ in range(epochs):

		# Forward pass
		z = features @ w + b
		y = 1/(1+np.exp(-z))

		# Loss
		mse = sum(((y-labels)**2))/n
		mse_values.append(mse)

		# Backward Pass
		w_grad = (2 * ((y-labels) * y * (1-y)) @ features)/n
		b_grad = (2 * sum((y-labels) * y * (1-y)))/n
	
		# Parameter Update
		w -= learning_rate * w_grad
		b -= learning_rate * b_grad

	return w, b, mse_values