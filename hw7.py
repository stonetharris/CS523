import numpy as np

def rmsnorm(input_tensor, weights):
    rms_value = np.sqrt(np.mean(input_tensor**2))
    normalized_tensor = input_tensor / rms_value
    weighted_normalized_tensor = weights * normalized_tensor
    return weighted_normalized_tensor

# input_a = np.array([5, 8, 7, 9, -2, 1, 0, 3])
# weights_g = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.35, 0, 0.05])
# input_a = np.array([1, 2, 3, 4, 5])
# weights_g = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

input_a = np.array([5, 8, 7, 9, -2, 1, 0, 3]) + np.array([5, 5, 5, 5, 5, 5, 5, 5])
weights_g = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.35, 0, 0.05])

output = rmsnorm(input_a, weights_g)
print(output)

