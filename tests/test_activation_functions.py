import pytest
import numpy as np

from tetris.agent.activation_functions import (
    sigmoid_activation_func,
    linear_activation_func,
    relu_activation_func,
    tanh_activation_func,
)


@pytest.fixture
def x() -> np.ndarray:
    return np.asarray([[0, 1], [-2, 3]])


def test_sigmoid_activation(x) -> None:
    true_x = np.asarray([[0.5, 0.7310585786300049], [0.11920292202211755, 0.9525741268224334]])
    np.testing.assert_array_equal(sigmoid_activation_func(x), true_x)


def test_linear_activation(x) -> None:
    np.testing.assert_array_equal(x, linear_activation_func(x))


def test_relu_activation(x) -> None:
    true_x = np.asarray([[0, 1], [0, 3]])
    np.testing.assert_array_equal(true_x, relu_activation_func(x))


def test_tanh_activation(x) -> None:
    true_x = np.asarray([[0, 0.7615941559557649], [-0.964027580075817, 0.9950547536867306]])
    np.testing.assert_array_equal(true_x, tanh_activation_func(x))
