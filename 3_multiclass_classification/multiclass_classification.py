'''
Numpy implementation of a multilayer perceptron 
with 3 layers such as: 
- 2 neurons in the input layer (2 features)
- 5 neurons in the hidden layer
- 3 neurons in the output layer (multiclass classification)
'''

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def cost_function(y, yhat):
    """ Cross-entropy cost function (y and yhat are arrays) """
    return np.sum(-Y * np.log(yhat))

def create_dataset(seed_nb):
    """ 
    Create dataset of 1500 observations clustered in 3 Gaussian clouds.
    Each observation has two features. The data is randomly generated based on the seed seed_nb.
    """
    np.random.seed(seed_nb)
    # generate three Gaussian clouds each holding 500 points
    X1 = np.random.randn(500, 2) + np.array([0, -2])   # (500 X 2)
    X2 = np.random.randn(500, 2) + np.array([2, 2])    # (500 X 2)
    X3 = np.random.randn(500, 2) + np.array([-2, 2])   # (500 X 2)

    # put them all in a big matrix
    X = np.vstack([X1, X2, X3])  # (1500 X 2)

    # generate the one-hot-encodings output array
    labels = np.array([0]*500 + [1]*500 + [2]*500)  # (1500 X 1)
    Y = np.zeros((1500, 3))
    for i in range(1500):
        Y[i, labels[i]] = 1    # (1500 X 3)

    return X, Y

# Create train and test data
X, Y = create_dataset(seed_nb=4526)
X_test, Y_test = create_dataset(seed_nb=7516)

# HYPERPARAMETERS
alpha = 10e-6
samples = X.shape[0] # 1500 samples
features = X.shape[1] # 2 features
hidden_nodes = 4
classes = 3
nb_epoch = 10000

# randomly initialize weights
W1 = np.random.randn(features, hidden_nodes)  # (2 X 5)
b1 = np.random.randn(hidden_nodes)            # (5 X 1)
W2 = np.random.randn(hidden_nodes, classes)   # (5 X 3)
b2 = np.random.randn(classes)                 # (3 X 1)

# TRAINING
costs = []

for epoch in range(nb_epoch):
    # Feedforward
    ## Layer 1
    Z1 = X.dot(W1) + b1  # (1500 X 5)
    A1 = sigmoid(Z1)  # (1500 X 5)

    ## Layer 2
    Z2 = A1.dot(W2) + b2 # (1500 X 3)
    A2 = softmax(Z2) # (1500 X 3)

    # cost function: cross-entropy
    J = cost_function(Y, A2)
    costs.append(J)

    # Backpropagation
    delta2 = A2 - Y              # (1500 X 3)
    delta1 = (delta2).dot(W2.T) * A1 * (1 - A1)    # (1500 X 5)

    # Layer 2
    W2 -= alpha * A1.T.dot(delta2)
    b2 -= alpha * (delta2).sum(axis=0)

    # Layer 1
    W1 -= alpha * X.T.dot(delta1)
    b1 -= alpha * (delta1).sum(axis=0)

    print("Epoch {}/{} | cost : {}".format(epoch, nb_epoch, J))

# TESTING
# Feedforward
Z1 = X_test.dot(W1) + b1  # (1500 X 5)
A1 = sigmoid(Z1)  # (1500 X 5)

Z2 = A1.dot(W2) + b2 # (1500 X 3)
A2 = softmax(Z2) # (1500 X 3)

Y_hat = A2.round()

# calculate test error
J_test = cost_function(Y_test, A2)
print('Train error final: ', J)
print('Test error final: ', J_test)

# PLOTS
plt.plot(costs)
plt.xlabel('Epoch #')
plt.ylabel('Training error')
plt.savefig('plots/3_J_vs_epoch.png')
plt.show()

plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.rainbow, label="Train set") 
plt.scatter(X_test[:,0], X_test[:,1], c=Y_hat, cmap=plt.cm.rainbow, marker='x', label="Test set")
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend()
plt.savefig('plots/3_test_vs_train.png')
plt.show()
