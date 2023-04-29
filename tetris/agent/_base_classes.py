from abc import ABC, abstractmethod


class BasePopulation(ABC):
    @abstractmethod
    def _crossover():
        pass

    @abstractmethod
    def _mutate():
        pass

    @abstractmethod
    def evolve():
        pass

    @abstractmethod
    def reset_scores():
        pass

    @abstractmethod
    def save_best_member():
        pass


class BaseAgent(ABC):
    @abstractmethod
    def evaluate_state():
        pass

    @abstractmethod
    def mutate():
        pass

    @abstractmethod
    def _save():
        pass
