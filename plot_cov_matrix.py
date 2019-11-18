import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')
import GPy

RANGE = [-4, 4]
LIN_NUM = 250

X = np.linspace(RANGE[0], RANGE[1], LIN_NUM)[:, None]
# Y = np.tanh(X[:,0]).reshape(-1,1)
Y = np.array([[0.]])
vs_ = [0., 1., 10.]
vs = np.linspace(0.1, 1, 1000)

k = GPy.kern.RBF(1)
k_R = k
C_R = k_R.K(X, np.array([[0.]]))
k_M = GPy.kern.Matern52(1)
C_M = k_M.K(X, np.array([[0.]]))
k_C = GPy.kern.Cosine(1)
C_C = k_C.K(X, np.array([[0.]]))

def plot_different_kernels(X, C_R, C_M, C_C):
    plt.figure(figsize=(18,7))
    plt.plot(X, C_R, X, C_M, X, C_C);
    plt.xlabel("x"), plt.ylabel("$\kappa$")
    plt.legend(labels=["Gaussian RBF", "Matérn 5/2", "Cosine"]);
    plt.show()

plot_different_kernels(X, C_R, C_M, C_C)
print(k)


def plot_kernel(figx, figy, X,Y):
    plt.figure(figsize=(figx, figy))
    #
    plt.subplot(121)
    K = k.K(X, Y)
    plt.plot(X, K)
    plt.title("x"), plt.ylabel("$\kappa$")
    plt.title("$\kappa_{rbf}(x,0)$")
    #
    plt.subplot(122)
    K = k.K(X,X)
    plt.pcolor(X.T, X, K)
    plt.gca().invert_yaxis(), plt.gca().axis("image")
    plt.xlabel("x"), plt.ylabel("x'"), plt.colorbar()
    plt.title("$\kappa_{rbf}(x,x')$")
    plt.show()

plot_kernel(12,4,X, Y)

def plot_variance(vs):
    plt.figure(figsize=(18, 7))
    for v in vs:
        k.variance = v
        C = k.K(X, np.array([[0.]]))
        plt.plot(X,C)
    plt.show()