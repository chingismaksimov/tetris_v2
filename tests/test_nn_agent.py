import pytest
import numpy as np
from tetris.agent.agent import NNAgent
from tetris.agent.population import NNAgentPopulation


@pytest.fixture(params=["linear", "relu"])
def agent(request) -> NNAgent:
    w = [
        np.asarray(
            [
                [1, 1, 1],
                [0, 1, 2],
                [3, 4, 5],
            ]
        ),
        np.asarray(
            [
                [0],
                [1],
                [-2],
            ]
        ),
    ]
    agent = NNAgent(w, request.param)
    return agent


@pytest.fixture
def population() -> NNAgentPopulation:
    return NNAgentPopulation(dim=[2, 3])


class TestEvaluateState:
    """
    A class to test `evaluate_state` method of `NNAgent` class.
    """

    @pytest.mark.parametrize("dim", [1, 3, 4])
    def test_incompatible_state_shape(self, agent, dim) -> None:
        state = np.random.normal(0, 1, dim)
        with pytest.raises(RuntimeError):
            agent.evaluate_state(state)

    def test_implementation(self, agent) -> None:
        state = np.asarray([-1, 1])
        if agent._activation == "linear":
            assert agent.evaluate_state(state) == -4
        else:
            assert agent.evaluate_state(state) == 0


class TestMutate:
    """
    A class to test `_mutate` method of `NNAgent` class.
    """

    def test_implementation(self, agent) -> None:
        true_weights = agent.w.copy()
        np.random.seed(4)
        true_weights = [w + np.random.normal(4, 3, w.shape) for w in true_weights]
        np.random.seed(4)
        agent._mutate(m=4, std=3)
        for i in range(len(true_weights)):
            np.testing.assert_array_equal(true_weights[i], agent.w[i])


class TestCrossover:
    """
    A class to test `_crossover` method of `NNAgentPopulation` class.
    """

    def test_implementation(self, population, agent) -> None:
        crossed_over_agent = population._crossover(agent, agent)
        for i in range(len(agent.w)):
            np.testing.assert_array_equal(agent.w[i], crossed_over_agent.w[i])
