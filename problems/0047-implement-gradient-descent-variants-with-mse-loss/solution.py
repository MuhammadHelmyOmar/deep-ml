import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    sample_size = len(X)
    w = weights.astype(float)

    if method == 'batch':
        batch_size = sample_size
    elif method == 'stochastic':
        batch_size = 1

    for _ in range(n_epochs):
        
        for batch_start in range(0, sample_size, batch_size):
            X_batch = X[batch_start : batch_start+batch_size]
            y_batch = y[batch_start : batch_start+batch_size]

            pred = X_batch @ w
            grad = (2/len(X_batch)) * (pred - y_batch) @ X_batch
            w -= learning_rate * grad
    
    return w