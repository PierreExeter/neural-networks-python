'''
Numpy implementation of a single-layer perceptron
with 3 neurons in the input layer (3 features)
and 1 output neuron (binary classification)
'''

from matplotlib import pyplot as plt  
import numpy as np  


def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) * (1-sigmoid(x))


# CREATE DATA SET
X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])  
y = np.array([[1, 0, 0, 1, 1]])  
y = y.reshape(5, 1)  

# HYPERPARAMETERS
np.random.seed(42)  
weights = np.random.rand(3, 1)  
bias = np.random.rand(1)  
alpha = 0.05        # learning rate
nb_epoch = 20000
H = np.zeros((nb_epoch, 6))  # history
m = len(X)   # number of observations

# TRAINING
for epoch in range(nb_epoch):  

    # FEEDFORWARD
    z = np.dot(X, weights) + bias  # (5x1)
    a = sigmoid(z)  # (5X1)

    # BACKPROPAGATION
    # 1. cost function: MSE
    J = (1/m) * (a - y)**2   # (5X1)

    # 2. weights
    dJ_da = (2/m)*(a-y) 
    da_dz = sigmoid_der(z)
    dz_dw = X.T

    gradient_w = np.dot(dz_dw, da_dz*dJ_da)  # chain rule 
    weights -= alpha*gradient_w               # gradient descent

    # 3. bias
    gradient_b = da_dz*dJ_da   # chain rule
    bias -= alpha*sum(gradient_b)  # gradient descent

    # Record history for plotting
    H[epoch, 0] = epoch
    H[epoch, 1] = J.sum()
    H[epoch, 2:5] = np.ravel(weights)
    H[epoch, 5] = np.asscalar(bias)

    print("Epoch {}/{} | cost : {}".format(epoch, nb_epoch, J.sum()))

# PLOT 
plt.plot(H[:, 0], H[:, 1])
plt.xlabel('Epoch #')
plt.ylabel('Training error')
plt.savefig('plots/1_J_vs_epoch.png')
plt.show()

plt.plot(H[:, 0], H[:, 2], label='$w_{11}$')
plt.plot(H[:, 0], H[:, 3], label='$w_{21}$')
plt.plot(H[:, 0], H[:, 4], label='$w_{31}$')
plt.xlabel('Epoch #')
plt.ylabel('Weights')
plt.legend()
plt.savefig('plots/1_weights_vs_epoch.png')
plt.show()

plt.plot(H[:, 0], H[:, 5])
plt.xlabel('Epoch #')
plt.ylabel('Bias')
plt.savefig('plots/1_bias_vs_epoch.png')
plt.show()

# TEST PHASE
example1 = np.array([1, 0, 1])  
result1 = sigmoid(np.dot(example1, weights) + bias)  
print(result1.round())
print('A person who is smoking, not obese and practices some exercise is classified as not diabetic.')

example2 = np.array([0, 1, 1])  
result2 = sigmoid(np.dot(example2, weights) + bias)  
print(result2.round())
print('A person who is not smoking, obese and practices some exercise is classified as diabetic.')
