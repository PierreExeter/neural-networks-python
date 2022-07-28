'''
Numpy implementation of a multilayer perceptron with 
2 neurons in the input layer (2 features)
4 neurons in the hidden layer
1 output neuron (binary classification)
'''

from sklearn import datasets  
import numpy as np  
import matplotlib.pyplot as plt


def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) * (1-sigmoid (x))


# CREATE TRAIN AND TEST DATA SET
np.random.seed(0)  
X, y = datasets.make_moons(500, noise=0.10)  
yy = y
y = y.reshape(500, 1)
X_test, y_test = datasets.make_moons(500, noise=0.10)  
y_test = y_test.reshape(500, 1)

# HYPERPARAMETERS
wh = np.random.rand(len(X[0]), 4)  
wo = np.random.rand(4, 1)  
alpha = 0.5  # learning rate
nb_epoch = 5000
error_list = []
H = np.zeros((nb_epoch, 14))  # history
m = X.shape[0] # number of observations

# TRAINING
for epoch in range(nb_epoch):  

    # 1. feedforward between input and hidden layer
    zh = np.dot(X, wh)
    ah = sigmoid(zh)

    # 2. feedforward between hidden and output layer
    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # 3. cost function: MSE
    J = (1/m) * (ao - y)**2 

    # 4. backpropagation between output and hidden layer
    dJ_dao = (2/m)*(ao-y) 
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah

    dJ_wo = np.dot(dzo_dwo.T, dJ_dao * dao_dzo)  # chain rule

    # 5. backpropagation between hidden and input layer
    dJ_dzo = dJ_dao * dao_dzo
    dzo_dah = wo
    dJ_dah = np.dot(dJ_dzo, dzo_dah.T)
    dah_dzh = sigmoid_der(zh) 
    dzh_dwh = X

    dJ_wh = np.dot(dzh_dwh.T, dah_dzh * dJ_dah)  # chain rule

    # 6. update weights: gradient descent (only at the end)
    wh -= alpha * dJ_wh
    wo -= alpha * dJ_wo

    # 7. record history for plotting
    H[epoch, 0] = epoch
    H[epoch, 1] = J.sum()
    H[epoch, 2:10] = np.ravel(wh)
    H[epoch, 10:14] = np.ravel(wo)

    print("Epoch {}/{} | cost : {}".format(epoch, nb_epoch, J.sum()))

# TESTING
zh = np.dot(X_test, wh)
ah = sigmoid(zh)
zo = np.dot(ah, wo)
ao = sigmoid(zo)

y_hat = ao.round()
y_hat = y_hat.reshape(500,)

J_test = (1/m)*(ao - y_test)**2 
print('Train error final: ', J.sum())
print('Test error final: ', J_test.sum())

# PLOT 
plt.plot(H[:, 0], H[:, 1])
plt.xlabel('Epoch #')
plt.ylabel('Training error')
plt.savefig('plots/2_J_vs_epoch.png')
plt.show()

plt.plot(H[:, 0], H[:, 2], label='$w_1$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 3], label='$w_2$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 4], label='$w_3$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 5], label='$w_4$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 6], label='$w_5$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 7], label='$w_6$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 8], label='$w_7$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 9], label='$w_8$', marker='x', markevery=200)
plt.plot(H[:, 0], H[:, 10], label='$w_9$', marker='o', markevery=200)
plt.plot(H[:, 0], H[:, 11], label='$w_{10}$', marker='o', markevery=200)
plt.plot(H[:, 0], H[:, 12], label='$w_{11}$', marker='o', markevery=200)
plt.plot(H[:, 0], H[:, 13], label='$w_{12}$', marker='o', markevery=200)
plt.xlabel('Epoch #')
plt.ylabel('Weights')
plt.legend()
plt.savefig('plots/2_Weights_vs_epoch.png')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=yy, cmap=plt.cm.Spectral, label="Train set") 
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_hat, cmap=plt.cm.Spectral, marker='x', label="Test set")
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.legend()
plt.savefig('plots/2_test_vs_train.png')
plt.show()
