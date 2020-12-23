import numpy as np
import matplotlib.pyplot as plt

def softplus(x):
    return np.log(1.0 + np.exp(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def mish(x):
    return x * tanh(softplus(x))

def swish(x):
    return x / (1 + np.exp(-x))

x = np.linspace(-4, 4, 1000)
softplus_y = softplus(x)
tanh_y = tanh(x)
mish_y = mish(x)
swish_y = swish(x)

plt.plot(x, softplus_y, label="softplus")
plt.plot(x, tanh_y, label="tanh")
plt.plot(x, mish_y, label="mish")
plt.plot(x, swish_y, label="swish")
plt.xlabel("x")
plt.ylabel("y")
plt.title("acrtivations")
plt.legend()
plt.savefig("./activation.png")