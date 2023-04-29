from tetris.environment._constants import (
    tetrominos,
    tetromino_starting_x,
    tetromino_starting_y,
    tetromino_structures,
    tetromino_to_num,
    board_height,
    board_width,
    max_height,
    tetris_multiplier,
)
import numpy as np
import random
from typing import Tuple, List, Dict


class TetrisEnvironment:
    def __init__(self):
        self._tetromino = random.choice(tetrominos)
        self._next_tetromino = random.choice(tetrominos)
        self._rotation = 0
        self._tetromino_x = tetromino_starting_x
        self._tetromino_y = tetromino_starting_y
        self._clear_board()

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def tetromino(self) -> str:
        return self._tetromino

    @property
    def next_tetromino(self) -> str:
        return self._next_tetromino

    @property
    def rotation(self) -> int:
        return self._rotation

    @rotation.setter
    def rotation(self, x: int) -> None:
        self._rotation = x

    @property
    def tetromino_x(self) -> int:
        return self._tetromino_x

    @tetromino_x.setter
    def tetromino_x(self, x: int) -> None:
        self._tetromino_x = x

    @property
    def tetromino_y(self) -> int:
        return self._tetromino_y

    @tetromino_y.setter
    def tetromino_y(self, x: int) -> int:
        self._tetromino_y = x

    def _clear_board(self) -> None:
        self._board = np.zeros((board_height, board_width))

    def return_board_representations(self) -> Dict[Tuple[int], List[np.ndarray]]:
        board_representations = {}
        for valid_placement in self._generate_valid_placements():
            board = self._place_tetromino_on_board(self._board, self._tetromino, *valid_placement)
            board_representations[valid_placement] = self._board_representation(board)
        return board_representations

    def _generate_valid_placements(self) -> Tuple[int]:
        """
        Generator of valid tetromino placements:
        - rotation;
        - x position;
        - y position.

        :yield: valid placement
        """
        col_heights = self._column_heights(self._board)
        for rotation in range(len(tetromino_structures[self._tetromino])):
            tetromino_structure = tetromino_structures[self._tetromino][rotation]
            tetromino_width = tetromino_structure.shape[1]
            tetromino_height = tetromino_structure.shape[0]
            for x in range(0, self._board.shape[1] - tetromino_width + 1):
                y = (
                    len(self._board)
                    - col_heights[x : x + tetromino_width]
                    - tetromino_height
                    + (tetromino_structure[::-1] != 0).argmax(axis=0)
                ).min()
                if y > tetromino_height:
                    yield (rotation, x, y)

    def _column_heights(self, board: np.ndarray) -> np.ndarray:
        column_heights = len(board) - (board != 0).argmax(axis=0)
        column_heights[board.sum(axis=0) == 0] = 0
        return column_heights

    def _place_tetromino_on_board(self, board: np.ndarray, tetromino: str, rotation: int, x: int, y: int) -> np.ndarray:
        """
        Places tetromino on board and returns updated board.

        :param board: board
        :param tetromino: tetromino
        :param rotation: current rotation of the tetromino
        :param x: x position of the tetromino
        :param y: y position of the tetromino
        :return: board with tetromino placed on it
        """
        board = board.copy()
        tetromino_structure = tetromino_structures[tetromino][rotation]
        tetromino_height, tetromino_width = tetromino_structure.shape
        if x < 0 or y < 0 or x + tetromino_width > board.shape[1] or y + tetromino_height > board.shape[0]:
            raise RuntimeError("Cannot place tetromino outside board.")

        for i in range(tetromino_height):
            for j in range(tetromino_width):
                if tetromino_structure[i, j] != 0:
                    if board[y + i][x + j] == 0:
                        board[y + i][x + j] = tetromino_to_num[tetromino]
                    else:
                        raise RuntimeError("Cannot place tetromino here.")
        return board

    def _board_representation(self, board: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                self._score_board_unevenness(board),
                self._score_n_holes(board),
                self._score_complete_lines(board),
                self._score_board_height(board),
            ]
        )

    def _score_board_unevenness(self, board: np.ndarray) -> float:
        return np.abs(np.diff(self._column_heights(board))).sum()

    def _score_n_holes(self, board: np.ndarray) -> float:
        return ((board == 0).sum(axis=0) - (len(board) - self._column_heights(board))).sum()

    def _score_complete_lines(self, board: np.ndarray) -> float:
        n_complete_lines = self._number_of_complete_lines(board)
        if n_complete_lines < 4:
            return n_complete_lines
        return n_complete_lines * tetris_multiplier

    def _number_of_complete_lines(self, board: np.ndarray) -> float:
        """
        Return number of complete lines.

        :param board: board
        :return: number of complete lines
        """
        return ((board != 0).sum(axis=1) == board.shape[1]).sum()

    def _score_board_height(self, board: np.ndarray) -> float:
        return self._column_heights(board).max()

    def process_best_placement(self, best_placement: Tuple[int], lines_cleared: int, games_played: int):
        self._board = self._place_tetromino_on_board(self._board, self._tetromino, *best_placement)
        n_cleared_lines = self._number_of_complete_lines(self._board)

        if n_cleared_lines > 0:
            lines_cleared = lines_cleared + n_cleared_lines
            self._board = self._board[(self._board == 0).any(axis=1)]
            self._board = np.append(np.zeros((n_cleared_lines, board_width)), self._board, axis=0)

        if self._column_heights(self._board).max() >= max_height:
            games_played = games_played + 1
            self._clear_board()

        self._spawn_new_tetromino()
        return lines_cleared, games_played

    def _spawn_new_tetromino(self) -> None:
        """
        Spawn a new tetromino.
        """
        self._tetromino = self._next_tetromino
        self._next_tetromino = random.choice(tetrominos)
        self._rotation = 0
        self._tetromino_x = tetromino_starting_x
        self._tetromino_y = tetromino_starting_y


def main():
    env = TetrisEnvironment()
    print(env.tetromino)
    print(env.board)


if __name__ == "__main__":
    main()
