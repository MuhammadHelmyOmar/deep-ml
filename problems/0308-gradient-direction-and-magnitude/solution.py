import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	gradient = np.array(gradient)

	mag = np.sqrt(np.sum(gradient**2))
	direc= (gradient/mag) if mag else np.zeros_like(gradient)
	desc_direc = -direc

	return {
		"magnitude": mag,
		"direction": direc,
		"descent_direction": -direc
	}