from typing import Dict, Any
from tetris.utils import read_yaml_file
from tetris.project_root_path import project_root_path
from tetris.agent.population import LinearAgentPopulation, NNAgentPopulation
from tetris.agent._base_classes import BasePopulation
from tetris.environment.environment import TetrisEnvironment
from pathlib import Path


def train(train_config: Dict[str, Any]):
    model = train_config["model"]
    if model == "linear":
        population = LinearAgentPopulation
    elif model == "nn":
        population = NNAgentPopulation
    population = population(**train_config["model_parameters"][model])

    env = TetrisEnvironment()

    for epoch in range(train_config["n_epochs"]):
        for i, member in enumerate(population.members):
            games_played = 0
            lines_cleared = 0
            while games_played < train_config["games_per_member"]:
                board_representations = env.return_board_representations()
                scored_representations = {k: member.evaluate_state(v) for k, v in board_representations.items()}
                best_placement = max(scored_representations, key=scored_representations.get)
                lines_cleared, games_played = env.process_best_placement(best_placement, lines_cleared, games_played)
            population.scores[i] = lines_cleared

        population.evolve()
        track_progress(epoch, population)
        population.reset_scores()

    if train_config["save_best_member"]:
        population.save_best_member(train_config["saving_paths"][model])


def track_progress(epoch: int, population: BasePopulation) -> None:
    """
    Track training progress.

    :param epoch: epoch of training
    :param population: population of individuals
    """
    print(f"Epoch {epoch + 1} is over.")
    print(f"Average score: {population.scores.mean()}")
    print(f"Highest score: {population.max_score}")
    print(f"Best member weights: {population.best_member.w}")
    print("\n")


def main():
    train_config = read_yaml_file(project_root_path / Path("training/train_config.yaml"))
    train(train_config)


if __name__ == "__main__":
    main()
