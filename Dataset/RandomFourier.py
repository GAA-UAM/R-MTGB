
#%%

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 500
D = 2  # Two input features
W = np.random.randn(N, D)  # Random weights for the Fourier features
b = np.random.uniform(0, 2*np.pi, N)  # Random biases for the Fourier features
theta = np.random.randn(N)  # Random weights for each Fourier feature
alpha = 100.0
l = 10.0

def f(x):
    x = np.atleast_2d(x)
    return np.apply_along_axis(lambda x: np.sum(theta * np.sqrt(alpha / N) * np.cos(np.dot(W, x / l) + b)), 1, x)

def classify_output(output):
    threshold = 1 
    return np.where(output > threshold, 1, 0)

# Generate random input features
x_values = np.random.uniform(-3, 3, size=(500, 2))
y_values = classify_output(f(x_values))

plt.scatter(x_values[:, 0], x_values[:, 1], c=y_values, cmap='viridis')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter plot of f(x)')
plt.show()

# %%
