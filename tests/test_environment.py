import pytest
import numpy as np
import random
from tetris.environment.environment import TetrisEnvironment
from tetris.environment._constants import (
    tetromino_starting_x,
    tetromino_starting_y,
    tetrominos,
    board_height,
    board_width,
    tetris_multiplier,
    tetromino_structures,
    tetromino_to_num,
)


@pytest.fixture
def env() -> TetrisEnvironment:
    env = TetrisEnvironment()
    env._tetromino = "J"
    env._next_tetromino = "T"
    env._tetromino_x = 4
    env._tetromino_y = 4
    return env


@pytest.fixture
def full_board() -> np.ndarray:
    np.random.choice(4)
    board = np.random.uniform(1, 8, (board_height, board_width))
    return board


@pytest.fixture
def empty_board() -> np.ndarray:
    board = np.zeros((board_height, board_width))
    return board


class TestSpawnNewTetromino:
    """
    A class to test `_spawn_new_tetromino` method of `TetrisEnvironment` class.
    """

    def test_implementation(self, env) -> None:
        random.seed(4)
        env._spawn_new_tetromino()
        random.seed(4)
        assert env._tetromino == "T"
        assert env._next_tetromino == random.choice(tetrominos)
        assert env._tetromino_x == tetromino_starting_x
        assert env._tetromino_y == tetromino_starting_y


class TestNumberOfCompleteLines:
    """
    A class to test `_number_of_complete_lines` method of `TetrisEnvironment` class.
    """

    def test_no_complete_lines(self, env, full_board) -> None:
        full_board[:, 0] = 0
        assert env._number_of_complete_lines(full_board) == 0

    def test_implementation(self, env, empty_board) -> None:
        empty_board[0:2, :] = 1
        assert env._number_of_complete_lines(empty_board) == 2


class TestScoreCompleteLines:
    """
    A class to test `_score_complete_lines` method of `TetrisEnvironment` class.
    """

    def test_no_complete_lines(self, env, full_board) -> None:
        full_board[:, 0] = 0
        assert env._score_complete_lines(full_board) == 0

    def test_tetris(self, env, empty_board) -> None:
        empty_board[:4, :] = np.random.choice(list(range(1, len(tetrominos) + 1)))
        assert env._score_complete_lines(empty_board) == 4 * tetris_multiplier

    @pytest.mark.parametrize("n_complete_lines", list(range(1, 4)))
    def test_implementation(self, env, empty_board, n_complete_lines) -> None:
        empty_board[np.random.choice(list(range(board_height)), n_complete_lines, replace=False), :] = np.random.choice(
            list(range(1, len(tetrominos) + 1))
        )
        assert env._score_complete_lines(empty_board) == n_complete_lines


class TestScoreNHoles:
    """
    A class to test `_score_n_holes` method of `TetrisEnvironment`.
    """

    def test_no_holes(self, env, full_board) -> None:
        board = full_board.copy()
        for i in range(board_height):
            board[i, :] = 0
            assert env._score_n_holes(board) == 0

        board = full_board.copy()
        for i in range(board_height):
            board[i, 0 : int(board_width / 2)] = 0
            assert env._score_n_holes(board) == 0

    def test_implementation(self, env, full_board) -> None:
        full_board[board_height - 2 :, 0] = 0
        n_holes = np.zeros(board_width)
        n_holes[0] = 2
        assert env._score_n_holes(full_board) == n_holes.sum()


class TestScoreBoardUnevenness:
    """
    A class to test `_score_board_unevenness` method of `TetrisEnvironment` class.
    """

    def test_even_board(self, env, full_board) -> None:
        board = full_board.copy()
        for i in range(board_height):
            board[i, :] = 0
        assert env._score_board_unevenness(board) == 0

        board = full_board.copy()
        board[np.random.choice(list(range(board_height)), np.random.randint(0, board_height), replace=False), :] = 0
        assert env._score_board_unevenness(board) == 0

    def test_implementation(self, env, full_board) -> None:
        full_board[:2, 0] = 0
        assert env._score_board_unevenness(full_board) == 2

        full_board[:3, 2] = 0
        assert env._score_board_unevenness(full_board) == 8


class TestScoreBoardHeight:
    """
    A class to test `_score_board_height` method of `TetrisEnvironment` class.
    """

    def test_empty_board(self, env, empty_board) -> None:
        assert env._score_board_height(empty_board) == 0

    def test_full_board(self, env, full_board) -> None:
        assert env._score_board_height(full_board) == board_height

    def test_implementation(self, env, empty_board) -> None:
        empty_board[5, 0] = 1
        assert env._score_board_height(empty_board) == board_height - 5


class TestColumnHeights:
    """
    A class to test `_column_heights` method of `TetrisEnvironment` class.
    """

    def test_empty_board(self, env, empty_board) -> None:
        assert (env._column_heights(empty_board) == np.zeros(board_width)).all()

    def test_full_board(self, env, full_board) -> None:
        assert (env._column_heights(full_board) == np.full(board_width, board_height)).all()

    def test_implementation(self, env, full_board) -> None:
        full_board[:, 0] = 0
        full_board[1:, 1] = 0
        full_board[: board_height - 2, 2] = 0
        true_column_heights = np.full(board_width, board_height)
        true_column_heights[0] = 0
        true_column_heights[2] = 2
        assert (env._column_heights(full_board) == true_column_heights).all()


class TestClearBoard:
    """
    A class to test `_clear_board` method of `TetrisEnvironment` class.
    """

    def test_implementation(self, env, full_board, empty_board) -> None:
        env._board = full_board
        assert (env._board != empty_board).any()
        env._clear_board()
        assert (env._board == empty_board).all()


class TestBoardRepresentation:
    """
    A class to test `_board_representation` method of `TetrisEnvironment` class.
    """

    def test_empty_board_representation(self, env, empty_board) -> None:
        true_representation = np.zeros(4)
        assert (env._board_representation(empty_board) == true_representation).all()


class TestPlaceTetrominoOnBoard:
    """
    A class to test `_place_tetromino_on_board`.
    """

    @pytest.mark.parametrize("tetromino", tetrominos)
    def test_placement_on_empty_board(self, env, empty_board, tetromino) -> None:
        for rotation in range(len(tetromino_structures[tetromino])):
            true_board_with_tetromino = empty_board.copy()
            tetromino_structure = tetromino_structures[tetromino][rotation]
            for y in range(tetromino_structure.shape[0]):
                for x in range(tetromino_structure.shape[1]):
                    if tetromino_structure[y][x] != 0:
                        true_board_with_tetromino[y + 5][x + 5] = tetromino_to_num[tetromino]
            assert (
                env._place_tetromino_on_board(empty_board, tetromino, rotation, 5, 5) == true_board_with_tetromino
            ).all()

    @pytest.mark.parametrize("tetromino", tetrominos)
    def test_placement_outside_board(self, env, empty_board, tetromino) -> None:
        with pytest.raises(RuntimeError):
            env._place_tetromino_on_board(empty_board, tetromino, 0, -1, 0)

        with pytest.raises(RuntimeError):
            env._place_tetromino_on_board(empty_board, tetromino, 0, 0, -1)

        with pytest.raises(RuntimeError):
            env._place_tetromino_on_board(empty_board, tetromino, 0, board_width, 0)

        with pytest.raises(RuntimeError):
            env._place_tetromino_on_board(empty_board, tetromino, 0, 0, board_height)

    @pytest.mark.parametrize("tetromino", tetrominos)
    def test_placement_on_non_empty_area(self, env, empty_board, tetromino) -> None:
        empty_board[5:9, 5:9] = 1
        for rotation in range(len(tetromino_structures[tetromino])):
            with pytest.raises(RuntimeError):
                env._place_tetromino_on_board(empty_board, tetromino, rotation, 5, 5)
