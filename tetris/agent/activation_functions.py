import numpy as np


def sigmoid_activation_func(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def linear_activation_func(x: np.ndarray) -> np.ndarray:
    return x


def relu_activation_func(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)


def tanh_activation_func(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


activation_functions = {
    "sigmoid": sigmoid_activation_func,
    "linear": linear_activation_func,
    "relu": relu_activation_func,
    "tanh": tanh_activation_func,
}
