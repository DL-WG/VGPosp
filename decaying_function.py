import numpy as np
import matplotlib.pyplot as plt

def f(l0):
    y = 0
    nom = (2*np.pi - l0)*(1+(np.cos(l0)/2)) + (3/2)*(np.sin(l0))
    if l0 < 2*(np.pi):
        y = l0 * nom / 3*np.pi
    return y

def g(l0, beta):
    ret = np.exp(-beta*l0**2 / (2*np.pi))
    return ret


def call_plot(l, beta):
    j = np.zeros_like(l)
    for i in range(len(l)):
        j[i] = g(l[i], beta)
    return j


if __name__ == '__main__':
    beta = 5
    l = np.linspace(0, 3 * np.pi, 100)

    v1 = [0.2, 0.5, 1, 2, 4, 6, 9, 15, 20]
    label = ["beta={}".format(v) for v in v1]
    for i in range(len(v1)):
        j = call_plot(l, v1[i])
        plt.plot(l, j, label=label[i])
    leg = plt.legend(loc='upper right')
    plt.xlabel("distance by index")
    plt.ylabel("decay function scaled by beta")
    plt.title("cut-off distance wrt beta")
    plt.show()

    pass