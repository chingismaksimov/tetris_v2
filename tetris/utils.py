import pygame
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from tetris.environment._constants import tetromino_colors, colors, num_to_tetromino, tetromino_structures


def read_yaml_file(path_to_file: Path) -> Dict[str, Any]:
    """
    Read yaml file and return as Python dictionary.

    :param path_to_file: path to yaml file
    :return: yaml file as Python dict
    """
    with open(path_to_file, "r") as file:
        return yaml.safe_load(file)


def initialize_pygame(pygame_config: Dict[str, Any]) -> Tuple[Any]:
    """
    Initialize pygame.

    :param pygame_config: pygame configurations
    :return: pygame display and clock objects
    """
    pygame.init()
    rectangle_size = pygame_config["rectangle_size"]
    display = pygame.display.set_mode(
        (pygame_config["display_width"] * rectangle_size, pygame_config["display_height"] * rectangle_size)
    )
    pygame.display.set_caption(pygame_config["caption"])
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(pygame_config["font"], pygame_config["font_size"])
    return display, clock, pygame_config["fps"], rectangle_size, font


def draw(
    display,
    board: np.ndarray,
    tetromino: str,
    rotation: int,
    tetromino_x: int,
    tetromino_y: int,
    rectangle_size: int,
    next_tetromino: str,
    lines_cleared: int,
    games_played: int,
    font,
) -> None:
    """
    Draw all the elements.

    :param display: pygame display
    :param board: board
    :param tetromino: current tetromino
    :param rotation: rotation of the current tetromino
    :param tetromino_x: x position of the current tetromino
    :param tetromino_y: y position of the current tetromino
    :param rectangle_size: rectangle size
    :param next_tetromino: next tetromino
    :param lines_cleared: number of lines cleared
    :param games_played: number of games played
    :param font: pygame font object
    """
    display.fill(colors["black"])
    draw_board(display, board, rectangle_size)
    draw_board_edge(display, board, rectangle_size)
    draw_tetromino(display, tetromino, rotation, tetromino_x, tetromino_y, rectangle_size)
    draw_tetromino(display, tetromino=next_tetromino, rotation=0, x=23, y=2, rectangle_size=rectangle_size)
    display.blit(
        *generate_text("Next piece:", font, x=18, y=5, color=colors["white"], rectangle_size=rectangle_size),
    )
    display.blit(
        *generate_text(
            f"Lines cleared: {lines_cleared}", font, x=20, y=10, color=colors["white"], rectangle_size=rectangle_size
        ),
    )
    display.blit(
        *generate_text(
            f"Games played: {games_played}", font, x=20, y=15, color=colors["white"], rectangle_size=rectangle_size
        ),
    )
    pygame.display.flip()


def generate_text(text: str, font, x: int, y: int, color, rectangle_size: int) -> Tuple[Any]:
    """
    Return pygame text object and the associated text rect object.

    :param text: text to render
    :param font: pygame font object
    :param x: x position of text
    :param y: y position of text
    :param color: text color
    :param rectangle_size: rectangle_size
    :return: pygame text object and the corresponding text rectangle
    """
    text = font.render(text, False, color)
    text_rect = text.get_rect()
    text_rect.center = (x * rectangle_size, y * rectangle_size)
    return text, text_rect


def draw_board(display, board: np.ndarray, rectangle_size: int) -> None:
    """
    Draw board.

    :param display: pygame display
    :param board: board to draw
    :param rectangle_size: rectangle size
    """
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            pos_value = board[row][col]
            if pos_value != 0:
                color = tetromino_colors[num_to_tetromino[pos_value]]
                pygame.draw.rect(
                    display,
                    color,
                    (
                        rectangle_size + col * rectangle_size,
                        rectangle_size + row * rectangle_size,
                        rectangle_size,
                        rectangle_size,
                    ),
                )
                pygame.draw.rect(
                    display,
                    colors["black"],
                    (
                        rectangle_size + col * rectangle_size,
                        rectangle_size + row * rectangle_size,
                        rectangle_size,
                        rectangle_size,
                    ),
                    1,
                )


def draw_board_edge(display, board: np.ndarray, rectangle_size: int) -> None:
    """
    Draw board edge.

    :param display: pygame display
    :param board: board
    :param rectangle_size: rectangle size
    """
    pygame.draw.rect(display, colors["grey"], (0, 0, (board.shape[1] + 2) * rectangle_size, rectangle_size))
    pygame.draw.rect(display, colors["grey"], (0, 0, rectangle_size, (board.shape[0] + 2) * rectangle_size))
    pygame.draw.rect(
        display,
        colors["grey"],
        (0, (len(board) + 1) * rectangle_size, (board.shape[1] + 2) * rectangle_size, rectangle_size),
    )
    pygame.draw.rect(
        display,
        colors["grey"],
        ((board.shape[1] + 1) * rectangle_size, 0, rectangle_size, (board.shape[0] + 2) * rectangle_size),
    )


def draw_tetromino(display, tetromino: str, rotation: int, x: int, y: int, rectangle_size: int) -> None:
    """
    Draw tetromino.

    :param display: pygame display
    :param tetromino: tetromino
    :param rotation: rotation of the tetromino
    :param x: x position of the tetromino
    :param y: y position of the tetromino
    :param rectangle_size: rectangle size
    """
    color = tetromino_colors[tetromino]
    tetromino_structure = tetromino_structures[tetromino][rotation]
    for row in range(len(tetromino_structure)):
        for col in range(tetromino_structure.shape[1]):
            if tetromino_structure[row][col] != 0:
                pygame.draw.rect(
                    display,
                    color,
                    ((1 + x + col) * rectangle_size, (1 + y + row) * rectangle_size, rectangle_size, rectangle_size),
                )
                pygame.draw.rect(
                    display,
                    colors["black"],
                    ((1 + x + col) * rectangle_size, (1 + y + row) * rectangle_size, rectangle_size, rectangle_size),
                    1,
                )
