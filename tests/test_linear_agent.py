import pytest
import numpy as np
from tetris.agent.agent import LinearAgent
from tetris.agent.population import LinearAgentPopulation


@pytest.fixture
def linear_agent() -> LinearAgent:
    return LinearAgent(np.arange(3))


class TestEvaluateState:
    """
    A class to test `evaluate_state` method of `LinearAgent` class.
    """

    def test_evaluate_state(self, linear_agent) -> None:
        assert linear_agent.evaluate_state(np.arange(3)) == 5

    def test_incompatible_state_shape(self, linear_agent) -> None:
        with pytest.raises(RuntimeError):
            linear_agent.evaluate_state(np.arange(4))


class TestMutate:
    """
    A class to test `mutate` method of `LinearAgent` class.
    """

    @pytest.mark.parametrize("std", [-1, 0])
    def test_negative_std(self, linear_agent, std) -> None:
        with pytest.raises(ValueError):
            linear_agent._mutate(0, std)

    @pytest.mark.parametrize("m", [0, 1])
    @pytest.mark.parametrize("std", [1, 10])
    def test_implementation(self, linear_agent, m, std) -> None:
        np.random.seed(4)
        true_weights = np.arange(3) + np.random.normal(m, std, 3)
        np.random.seed(4)
        linear_agent._mutate(m, std)
        np.testing.assert_array_equal(true_weights, linear_agent.w)


class TestLinearAgentPopulationInitMethod:
    """
    A class to test `__init__` method of `LinearAgentPopulation` class.
    """

    @pytest.mark.parametrize("init_weights", [None, np.asarray([10, 10, 10])])
    def test_bias(self, init_weights) -> None:
        np.random.seed(4)
        linear_agent_population = LinearAgentPopulation(dim=3, init_weights=init_weights)
        if init_weights is None:
            init_weights = np.zeros(3)
        np.random.seed(4)
        for agent in linear_agent_population._members:
            np.testing.assert_array_equal(agent.w, init_weights + np.random.uniform(0, 1, 3))


@pytest.fixture(params=[False, True])
def linear_agent_population(request) -> LinearAgentPopulation:
    population = LinearAgentPopulation(dim=3, n_members=3, elitism=request.param)
    population._members[0].w = np.arange(3)
    population._members[1].w = np.arange(1, 4)
    population._members[2].w = np.arange(2, 5)
    population._scores = np.arange(3)
    return population


class TestCrossOverMethod:
    """
    A class to test `_crossover` method of `LinearAgentPopulation` class.
    """

    def test_implementation(self, linear_agent_population) -> None:
        crossed_member = linear_agent_population._crossover(
            linear_agent_population._members[0], linear_agent_population._members[1]
        )
        np.testing.assert_array_equal(crossed_member.w, (np.arange(3) + np.arange(1, 4)) / 2)


class TestMutateMethod:
    """
    A class to test `_mutate` method of `LinearAgentPopulation` class.
    """

    def test_implementation(self, linear_agent_population) -> None:
        np.random.seed(4)
        linear_agent_population._mutate()
        np.random.seed(4)
        for i, member in enumerate(linear_agent_population._members):
            if linear_agent_population._elitism and i == 2:
                np.testing.assert_array_equal(member.w, np.arange(2, 5))
            else:
                np.testing.assert_array_equal(member.w, np.arange(i, i + 3) + np.random.normal(0, 1, 3))


class TestEvolveMethod:
    """
    A class to test `evolve` method of `LinearAgentPopulation` class.
    """

    def test_negative_scores(self, linear_agent_population) -> None:
        linear_agent_population._scores = np.asarray([-1, 0, 2])
        with pytest.raises(ValueError):
            linear_agent_population.evolve()

    def test_implementation(self, linear_agent_population) -> None:
        np.random.seed(4)
        linear_agent_population.evolve()
        np.random.seed(4)
        member_pair_idx = [np.random.choice(np.arange(3), 2, p=np.arange(3) / np.arange(3).sum()) for i in range(3)]
        member_pairs = [
            (np.arange(1, 4) if idx[0] == 1 else np.arange(2, 5), np.arange(1, 4) if idx[1] == 1 else np.arange(2, 5))
            for idx in member_pair_idx
        ]
        for i, member in enumerate(linear_agent_population._members):
            if linear_agent_population._elitism and i == 2:
                np.testing.assert_array_equal(member.w, np.arange(2, 5))
            else:
                np.testing.assert_array_equal(
                    member.w, (member_pairs[i][0] + member_pairs[i][1]) / 2 + np.random.normal(0, 1, 3)
                )


class TestResetScores:
    """
    A class to test `reset_scores` method of `LinearAgentPopulation` class.
    """

    def test_implementation(self, linear_agent_population) -> None:
        linear_agent_population.reset_scores()
        np.testing.assert_array_equal(linear_agent_population.scores, np.zeros(3))
