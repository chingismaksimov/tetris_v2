from __future__ import annotations
from tetris.agent._base_classes import BaseAgent, BasePopulation
import numpy as np
from typing import Optional, List, Union
from tetris.project_root_path import project_root_path
from pathlib import Path


class LinearAgentPopulation(BasePopulation):
    def __init__(
        self,
        dim: int,
        n_members: int = 100,
        low: float = 0,
        high: float = 1,
        bias: Optional[Union[np.ndarray, List[float]]] = None,
        elitism: bool = True,
        m: float = 0,
        std: float = 1,
    ):
        """
        Instantiate `LinearAgentPopulation` class.

        :param dim: dimension of the weights
        :param n_members: number of members in the population
        :param low: the lowest value to sample from the uniform distribution when instantiating member weights
        :param high: the highest value to sample from the uniform distribution when instantiating member weights
        :param bias: bias to add to the weights when instantiating members of the population
        :param elitism: if True, keep the currently best member unchanged from the previous population
        :param m: mean of the Gaussian noise applied as part of mutation
        :param std: standard deviation of the Gaussian noise applied as part of mutation
        """
        self._n_members = n_members
        self.reset_scores()
        if bias is None:
            bias = np.zeros(dim)
        else:
            bias = np.asarray(bias)
        if len(bias) != dim:
            raise RuntimeError("Provided `bias` is incompatible with the dimension.")
        self._members = [LinearAgent(w + bias) for w in np.random.uniform(low, high, (n_members, dim))]
        self._elitism = elitism
        self._m = m
        self._std = std
        self._best_member = None
        self._max_score = 0

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @property
    def members(self) -> List[LinearAgent]:
        return self._members

    @property
    def best_member(self) -> LinearAgent:
        return self._best_member

    @property
    def max_score(self) -> float:
        return self._max_score

    def reset_scores(self) -> None:
        """
        Reset scores.
        """
        self._scores = np.zeros(self._n_members)

    def evolve(self) -> None:
        """
        Evolve the population by applying crossover operator and mutation.
        """
        if (self._scores < 0).any():
            raise ValueError("Scores cannot be negative.")

        if np.max(self._scores) > self._max_score:
            self._best_member = self._members[np.argmax(self._scores)]
            self._max_score = np.max(self._scores)

        member_pair_idx = [
            np.random.choice(np.arange(self._n_members), 2, p=self._scores / self._scores.sum())
            for i in range(self._n_members)
        ]
        member_pairs = [(self._members[idx[0]], self._members[idx[1]]) for idx in member_pair_idx]

        if self._elitism:
            self._members = [self._members[np.argmax(self._scores)]] + [
                self._crossover(*pair) for pair in member_pairs[:-1]
            ]
        else:
            self._members = [self._crossover(*pair) for pair in member_pairs]

        self._mutate()

    def _crossover(self, member_1: LinearAgent, member_2: LinearAgent) -> None:
        """
        Apply crossover operator on two members of the population.

        :param member_1: first member of the population
        :param member_2: second member of the population
        :return: new member of the population as the result of the crossover operation
        """
        return LinearAgent((member_1.w + member_2.w) / 2)

    def _mutate(self) -> None:
        """
        Mutate members.
        """
        for i, member in enumerate(self._members):
            if self._elitism and i == np.argmax(self._scores):
                continue
            else:
                member.mutate(self._m, self._std)

    def save_best_member(self, file_name: str) -> None:
        """
        Save weights of the best performing member in the population.

        :param file_name: file name
        """
        self._best_member._save(file_name)


class LinearAgent(BaseAgent):
    def __init__(self, w: np.ndarray) -> LinearAgent:
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

    def mutate(self, m: float = 0, std: float = 1) -> None:
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
