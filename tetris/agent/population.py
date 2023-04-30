from tetris.agent._base_classes import BasePopulation
from tetris.agent.agent import LinearAgent, NNAgent
import numpy as np
from typing import Optional, Union, List


class Population(BasePopulation):
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

    def _mutate(self) -> None:
        """
        Mutate members.
        """
        for i, member in enumerate(self._members):
            if self._elitism and i == np.argmax(self._scores):
                continue
            else:
                member._mutate(self._m, self._std)

    def save_best_member(self, file_name: str) -> None:
        """
        Save weights of the best performing member in the population.

        :param file_name: file name
        """
        self._best_member._save(file_name)


class LinearAgentPopulation(Population):
    def __init__(
        self,
        dim: int,
        n_members: int = 100,
        low: float = 0,
        high: float = 1,
        init_weights: Optional[Union[np.ndarray, List[float]]] = None,
        elitism: bool = True,
        m: float = 0,
        std: float = 1,
    ):
        """
        :param dim: dimension of the weights
        :param n_members: number of members in the population
        :param low: the lowest value to sample from the uniform distribution when instantiating member weights
        :param high: the highest value to sample from the uniform distribution when instantiating member weights
        :param init_weights: initial_weights
        :param elitism: if True, keep the currently best member unchanged from the previous population
        :param m: mean of the Gaussian noise applied as part of mutation
        :param std: standard deviation of the Gaussian noise applied as part of mutation
        """
        self._n_members = n_members
        self.reset_scores()
        if init_weights is None:
            init_weights = np.zeros(dim)
        else:
            init_weights = np.asarray(init_weights)
        if len(init_weights) != dim:
            raise RuntimeError("Provided `init_weights` is incompatible with the dimension.")
        self._members = [LinearAgent(w + init_weights) for w in np.random.uniform(low, high, (n_members, dim))]
        self._elitism = elitism
        self._m = m
        self._std = std
        self._best_member = None
        self._max_score = 0

    def _crossover(self, member_1: LinearAgent, member_2: LinearAgent) -> None:
        """
        Apply crossover operator on two members of the population.

        :param member_1: first member of the population
        :param member_2: second member of the population
        :return: new member of the population as the result of the crossover operation
        """
        return LinearAgent((member_1.w + member_2.w) / 2)


class NNAgentPopulation(Population):
    def __init__(
        self,
        dim: List[int],
        n_members: int = 100,
        low: float = 0,
        high: float = 1,
        activation: str = "sigmoid",
        elitism: bool = True,
        m: float = 0,
        std: float = 1,
    ):
        """
        :param dim: numbers of neurons in input and hidden layers
        :param n_members: number of members in the population
        :param low: the lowest value to sample from the uniform distribution when instantiating member weights
        :param high: the highest value to sample from the uniform distribution when instantiating member weights
        :param activation: activation function
        :param elitism: if True, keep the currently best member unchanged from the previous population
        :param m: mean of the Gaussian noise applied as part of mutation
        :param std: standard deviation of the Gaussian noise applied as part of mutation
        """
        self._n_members = n_members
        self.reset_scores()
        dim = [d + 1 for d in dim]
        dim = dim + [1]
        self._members = [
            NNAgent(
                w=[np.random.uniform(low, high, (dim[i], dim[i + 1])) for i in range(len(dim) - 1)],
                activation=activation,
            )
            for member in range(n_members)
        ]
        self._activation = activation
        self._elitism = elitism
        self._m = m
        self._std = std
        self._best_member = None
        self._max_score = 0

    def _crossover(self, member_1: LinearAgent, member_2: LinearAgent) -> None:
        """
        Apply crossover operator on two members of the population.

        :param member_1: first member of the population
        :param member_2: second member of the population
        :return: new member of the population as the result of the crossover operation
        """
        return NNAgent(
            w=[(member_1.w[i] + member_2.w[i]) / 2 for i in range(len(member_1.w))],
            activation=self._activation,
        )
