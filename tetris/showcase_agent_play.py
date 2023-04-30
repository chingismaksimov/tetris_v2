import pygame
from pygame.locals import QUIT
from tetris.utils import read_yaml_file, initialize_pygame, draw
from tetris.project_root_path import project_root_path
from tetris.agent.agent import LinearAgent, NNAgent
from tetris.environment.environment import TetrisEnvironment
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Any
import sys


def showcase_agent_play(config: Dict[str, Any]) -> None:  # noqa
    trained_agent_path = project_root_path / Path(config["trained_agent_file_path"])
    if config["agent_type"] == "linear":
        agent = LinearAgent(np.load(trained_agent_path))
    elif config["agent_type"] == "nn":
        with open(trained_agent_path, "rb") as f:
            agent = NNAgent(*pickle.load(f).values())
    else:
        raise RuntimeError(f"Agent type '{agent}' not supported.")

    env = TetrisEnvironment()

    display, clock, fps, rectangle_size, font = initialize_pygame(config["pygame_config"])

    games_played = 0
    lines_cleared = 0

    while True:
        board_representations = env.return_board_representations()
        scored_representations = {k: agent.evaluate_state(v) for k, v in board_representations.items()}
        best_placement = max(scored_representations, key=scored_representations.get)
        target_rotation, target_x, target_y = best_placement
        while env.rotation != target_rotation or env.tetromino_x != target_x or env.tetromino_y != target_y:
            action = 0
            if action == 0 and env.rotation != target_rotation:
                env.rotation += 1
                action = 1
            if action == 0 and env.tetromino_x != target_x:
                if env.tetromino_x < target_x:
                    env.tetromino_x += 1
                else:
                    env.tetromino_x -= 1
                action = 1
            if action == 0 and env.tetromino_y != target_y:
                env.tetromino_y += 1
                action = 1
            draw(
                display=display,
                board=env.board,
                tetromino=env.tetromino,
                rotation=env.rotation,
                tetromino_x=env.tetromino_x,
                tetromino_y=env.tetromino_y,
                rectangle_size=rectangle_size,
                next_tetromino=env.next_tetromino,
                lines_cleared=lines_cleared,
                games_played=games_played,
                font=font,
            )
            clock.tick(fps)
        lines_cleared, games_played = env.process_best_placement(best_placement, lines_cleared, games_played)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


def main():
    config = read_yaml_file(project_root_path / Path("config.yaml"))
    showcase_agent_play(config)


if __name__ == "__main__":
    main()
