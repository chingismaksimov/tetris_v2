import numpy as np


board_width = 10
board_height = 20

max_height = board_height - 4  # max height before game is over
tetris_multiplier = 2  # multiplier for clearing 4 lines at once

tetromino_starting_x = int(board_width / 2)
tetromino_starting_y = 0

colors = {"black": (0, 0, 0), "white": (255, 255, 255), "grey": (47, 79, 79)}

tetrominos = ["T", "I", "J", "L", "O", "S", "Z"]
tetromino_to_num = dict(zip(tetrominos, range(1, len(tetrominos) + 1)))
num_to_tetromino = {v: k for k, v in tetromino_to_num.items()}

tetromino_colors = {
    "T": (255, 0, 0),
    "I": (0, 255, 255),
    "J": (0, 0, 255),
    "L": (0, 255, 0),
    "O": (255, 255, 0),
    "S": (255, 165, 0),
    "Z": (160, 32, 240),
}

tetromino_structures = {
    "T": [
        np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
            ]
        ),
        np.array(
            [
                [0, 1],
                [1, 1],
                [0, 1],
            ]
        ),
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
            ]
        ),
        np.array(
            [
                [1, 0],
                [1, 1],
                [1, 0],
            ]
        ),
    ],
    "I": [
        np.array(
            [
                [1, 1, 1, 1],
            ],
        ),
        np.array(
            [
                [1],
                [1],
                [1],
                [1],
            ]
        ),
    ],
    "J": [
        np.array(
            [
                [0, 1],
                [0, 1],
                [1, 1],
            ]
        ),
        np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
            ]
        ),
        np.array(
            [
                [1, 1],
                [1, 0],
                [1, 0],
            ]
        ),
        np.array(
            [
                [1, 1, 1],
                [0, 0, 1],
            ]
        ),
    ],
    "L": [
        np.array(
            [
                [1, 0],
                [1, 0],
                [1, 1],
            ]
        ),
        np.array(
            [
                [1, 1, 1],
                [1, 0, 0],
            ]
        ),
        np.array(
            [
                [1, 1],
                [0, 1],
                [0, 1],
            ]
        ),
        np.array(
            [
                [0, 0, 1],
                [1, 1, 1],
            ]
        ),
    ],
    "O": [
        np.array(
            [
                [1, 1],
                [1, 1],
            ]
        )
    ],
    "S": [
        np.array(
            [
                [0, 1, 1],
                [1, 1, 0],
            ]
        ),
        np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ]
        ),
    ],
    "Z": [
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
            ]
        ),
        np.array(
            [
                [0, 1],
                [1, 1],
                [1, 0],
            ]
        ),
    ],
}
