import numpy as np
from scipy.optimize import minimize

def relu(x):
    return np.maximum(0, x)

def reconstruct_x(a, W2):
    return relu(W2.T @ a)

def compute_a(x, W1):
    return relu(W1.T @ x)

# loss function that we want to minimize
def loss_function(params, x1, x2, x3):
    W1 = params[:12].reshape((4, 3)) 
    W2 = params[12:].reshape((3, 4)) 
    a1 = compute_a(x1, W1)
    a2 = compute_a(x2, W1)
    a3 = compute_a(x3, W1)
    
    # reconstruct
    x1_reconstructed = reconstruct_x(a1, W2)
    x2_reconstructed = reconstruct_x(a2, W2)
    x3_reconstructed = reconstruct_x(a3, W2)
    
    # compute loss as the sum of squared differences
    loss = np.sum((x1 - x1_reconstructed)**2)
    loss += np.sum((x2 - x2_reconstructed)**2)
    loss += np.sum((x3 - x3_reconstructed)**2)
    
    return loss

# initial guess
W1_initial = np.random.rand(4, 3)
W2_initial = np.random.rand(3, 4)
initial_params = np.concatenate((W1_initial.ravel(), W2_initial.ravel()))

# given x values
x1 = np.array([0, 1, 0, 1])
x2 = np.array([0, 0, 1, 1])
x3 = np.array([1, 0, 0, 0])

# perform optimization
result = minimize(loss_function, initial_params, args=(x1, x2, x3), method='BFGS')

if result.success:
    W1_optimized = result.x[:12].reshape((4, 3))
    W2_optimized = result.x[12:].reshape((3, 4))
    print("Optimized W1:", W1_optimized)
    print("Optimized W2:", W2_optimized)
else:
    print("Optimization failed:", result.message)