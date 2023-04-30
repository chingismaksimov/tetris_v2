from __future__ import annotations
from tetris.agent._base_classes import BaseAgent
from tetris.project_root_path import project_root_path
from tetris.agent.activation_functions import activation_functions
import numpy as np
from functools import reduce
import pickle
from pathlib import Path
from typing import List


class LinearAgent(BaseAgent):
    def __init__(self, w: np.ndarray) -> LinearAgent:
        """
        :param w: weights
        """
        self._w = w

    @property
    def w(self) -> np.ndarray:
        return self._w

    @w.setter
    def w(self, x: np.ndarray) -> None:
        self._w = x

    def evaluate_state(self, state: np.ndarray) -> float:
        """
        Evaluate input state.

        :param state: state to evaluate
        :return: evaluation of the input state
        """
        if self._w.shape != state.shape:
            raise RuntimeError("Provided state is incompatible with own weights.")
        return (self._w * state).sum()

    def _mutate(self, m: float = 0, std: float = 1) -> None:
        """
        Mutate weights by adding Gaussian noise.

        :param m: mean of the Gaussian noise
        :param std: standard deviation of the Gaussian noise
        """
        if std <= 0:
            raise ValueError("Standard deviation must be strictly positive.")
        self._w = self._w + np.random.normal(m, std, self._w.shape)

    def _save(self, file_name: str) -> None:
        """
        Save weights.

        :param file_name: file name
        """
        np.save(project_root_path / Path(file_name), self._w)


class NNAgent(BaseAgent):
    def __init__(self, w: List[np.ndarray], activation: str = "relu") -> NNAgent:
        """
        :param w: weights
        :param activation: activation function
        """
        self._w = w
        self._activation = activation

    @property
    def w(self) -> List[np.ndarray]:
        return self._w

    @w.setter
    def w(self, x: List[np.ndarray]) -> None:
        self._w = x

    def evaluate_state(self, state: np.ndarray) -> float:
        """
        Evaluate input state.

        :param state: state to evaluate
        :return: evaluation of the input state
        """
        state = np.insert(state, 0, 1)
        state = state.reshape(1, -1)

        activation_func = activation_functions[self._activation]

        if state.shape[1] != self._w[0].shape[0]:
            raise RuntimeError("Provided state is incompatible with own weights.")

        return float(reduce(lambda x, y: activation_func(x @ y), [state] + self._w))

    def _mutate(self, m: float = 0, std: float = 1) -> None:
        """
        Mutate weights by adding Gaussian noise.

        :param m: mean of the Gaussian noise
        :param std: standard deviation of the Gaussian noise
        """
        if std <= 0:
            raise ValueError("Standard deviation must be strictly positive.")
        self._w = [w + np.random.normal(m, std, w.shape) for w in self._w]

    def _save(self, file_name: str) -> None:
        """
        Save weights.

        :param file_name: file name
        """
        with open(project_root_path / Path(file_name + ".pkl"), "wb") as f:
            pickle.dump(self.__dict__, f)
