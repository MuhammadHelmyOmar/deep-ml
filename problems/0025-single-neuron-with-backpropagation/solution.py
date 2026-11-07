import numpy as np

def train_neuron(
	features: np.ndarray, 
	labels: np.ndarray, 
	initial_weights: np.ndarray, 
	initial_bias: float, 
	learning_rate: float, 
	epochs: int) -> (np.ndarray, float, list[float]):

	w, b = np.array(initial_weights, dtype=(float)), float(initial_bias)
	n = len(features)
	mse_values = []

	for _ in range(epochs):

		# Forward pass
		z = features @ w + b
		y = 1/(1+np.exp(-z))

		# Loss
		error = y-labels
		mse_values.append(np.mean(error**2))

		# Backward Pass
		delta = (2/n)* error * y * (1-y)
	
		# Parameter Update
		w -= learning_rate * (delta @ features)
		b -= learning_rate * delta.sum()

	return np.round(w, 4), np.round(b, 4), np.round(mse_values, 4).tolist()