'''
Implementation of a multilayer perceptron in Pytorch with 
- 2 neurons in the input layer (2 features)
- 4 neurons in the hidden layer
- 1 output neuron (binary classification)
'''

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler   
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# CREATE TRAIN AND TEST SET

def create_dataset(seed_nb):
    """ Define X and y """
    np.random.seed(seed_nb)
    X, y = make_moons(500, noise=0.10)  

    # Standardize the input
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Transform to tensor
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)

    # Reshape y
    new_shape = (len(y), 1)
    y = y.view(new_shape)

    return X, y

X, y = create_dataset(seed_nb=4564)
X_test, y_test = create_dataset(seed_nb=8472)

# HYPERPARAMETERS
alpha = 0.5  # learning rate
nb_epoch = 5000

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self):
        """
        Define 1 input layer of 2 neurons, 1 hidden layer of 2 neurons and 1 output layer of 2 neurons
        Applies a linear transformation between each layer z = W X + B with the Sigmoid activation function
        """
        super(Net, self).__init__()
        self.L0 = nn.Linear(2, 4)
        self.N0 = nn.Sigmoid()
        self.L1 = nn.Linear(4, 1)

    def forward(self, x):
        """ Feedforward from the input to the output layer """
        x = self.L0(x)
        x = self.N0(x)
        x = self.L1(x)
        return x
    
    def predict(self, x):
        """ Predict the class (0 or 1) of input x """
        return self.forward(x)


model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
print(model)

# TRAINING
costs = []

for epoch in range(nb_epoch):
    
    # Feedforward
    y_pred = model.forward(X)
    
    # Compute cost function
    cost = criterion(y_pred, y)

    # Add loss to the list
    costs.append(cost.item())

    # Clear the previous gradients
    optimizer.zero_grad()
    
    # Compute gradients
    cost.backward()
    
    # Adjust weights
    optimizer.step()
    
    if epoch % (nb_epoch // 100) == 0:
        print("Epoch {}/{} | cost : {:.3f}".format(epoch, nb_epoch, cost.item()))


# TESTING
y_pred_test = model.predict(X_test)
y_pred_test = torch.round(y_pred_test)
cost_test = criterion(y_pred_test, y)

print('Train error final: ', cost.item())
print('Test error final: ', cost_test.item())


# PLOT DECISION BOUNDARY
def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.detach().numpy()

def plot_decision_boundary(pred_func, X_train, y_train, X_test, y_test):

    # Define mesh grid
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Define figure and colormap
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = plt.cm.viridis
    first, last = cm.colors[0], cm.colors[-1]
    cm_bright = ListedColormap([first, last])

    # Plot the contour, decision boundary, test and train data
    cb = ax.contourf(xx, yy, Z, levels=10, cmap=cm, alpha=0.8)
    CS = ax.contour(xx, yy, Z, levels=[.5], colors='k', linestyles='dashed', linewidths=2)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', marker='o', s=100, linewidth=2, label="Train data")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k',marker='^', s=100, linewidth=2, label="Test data")

    # Colourbar, axis, title, legend
    fs = 15
    plt.clabel(CS, inline=1, fontsize=fs)
    CS.collections[0].set_label("Decision boundary at 0.5")
    plt.colorbar(cb, ticks=[0, 1])
    ax.legend(fontsize=fs, loc='upper left')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("$X_1$", fontsize=fs)
    ax.set_ylabel("$X_2$", fontsize=fs)
    ax.set_xticks(())
    ax.set_yticks(())


plot_decision_boundary(lambda x : predict(x), X.numpy(), y.numpy(), X_test.numpy(), y_test.numpy())
plt.tight_layout()
plt.savefig('plots/4_train_vs_test.png')
plt.show()


# PLOT COST FUNCTION VS EPOCHS
plt.plot(range(nb_epoch), costs)
plt.xlabel('Epoch #')
plt.ylabel('Training error')
plt.savefig('plots/4_J_vs_epoch.png')
plt.show()
