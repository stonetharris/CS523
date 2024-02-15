import numpy as np

# function f(theta) and its derivative df(theta)
def f(theta):
    return 2 * (theta - 2)**4

def df(theta):
    return 8 * (theta - 2)**3

# Given
theta = 4
learning_rate = 0.01

# One step of gradient descent
theta_new = theta - learning_rate * df(theta)

# Compute
f_updated = f(theta_new)
# print(f_updated)

# -----------------------------------------------------------------------

# Parameters for each layer
input_size = (10, 10, 3)  # (height, width, depth)

# First Conv Layer
conv1_kernels = 32
conv1_kernel_size = (5, 5, input_size[2])  # (height, width, depth)
conv1_params = (conv1_kernel_size[0] * conv1_kernel_size[1] * conv1_kernel_size[2] + 1) * conv1_kernels

# Second Conv Layer
# Depth of the output of the first conv layer becomes the depth for the second
conv2_kernels = 64
conv2_kernel_size = (5, 5, conv1_kernels)  # (height, width, depth)
conv2_params = (conv2_kernel_size[0] * conv2_kernel_size[1] * conv2_kernel_size[2] + 1) * conv2_kernels

# Fully-Connected Layers
fc_input_size = (input_size[0] // 2 // 2) * (input_size[1] // 2 // 2) * conv2_kernels
fc1_neurons = 128
fc1_params = fc_input_size * fc1_neurons + fc1_neurons  # weights + biases

fc2_neurons = 10
fc2_params = fc1_neurons * fc2_neurons + fc2_neurons  # weights + biases

# Sum all parameters
total_params = conv1_params + conv2_params + fc1_params + fc2_params
# print(total_params)

# -----------------------------------------------------------------------

# Given RNN parameters
W = np.array([[-1, 0], [0, -1]])
U = np.array([[1], [1]])
s_0 = np.array([[0], [0]])

# Input x
x = np.array([[1], [0]])

# First state calculation
s_1 = W.dot(s_0) + U * x  # Element-wise multiplication for U and x

# Second input is 0, hence U.x will be [0,0] for s_2
x = np.array([[0], [0]])

# Second state calculation
s_2 = W.dot(s_1)  # No need to add U.dot(x) since x is [0,0]

print(s_2)
