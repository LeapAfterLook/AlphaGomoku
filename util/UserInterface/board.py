import numpy as np
import matplotlib.pyplot as plt

import sys
if not ".." in sys.path:
    sys.path.append("..")
from error import ImpossiblePositionError, ImpossibleColorError, MoveFormatError, BoardDuplicateError


# index dictionary
dic_char_to_num = {}
dic_num_to_char = {}
char_list = ' '.join("abcdefghijklmno").split()
for i in range(0, 15):
    dic_char_to_num[char_list[i]] = i
    dic_num_to_char[i] = char_list[i]


def position_char_to_num(position_char):
    try:
        if len(position_char) > 3:
            raise ImpossiblePositionError("Too long position description")
        position_row_num = 15 - int(position_char[1:])
        position_col_num = dic_char_to_num[position_char[0]]
        return position_row_num, position_col_num
    except ImpossiblePositionError as e:
        print(e)
        exit(1)


def position_num_to_char(*args):
    try:
        if len(args) is 1:
            position_seq_num = args[0]
            position_row_num, position_col_num = position_seq_num // 15, position_seq_num % 15
        elif len(args) is 2:
            position_row_num, position_col_num = args
        else:
            raise ImpossiblePositionError("Wrong position description")
        if (0 <= position_row_num <= 14) and (0 <= position_col_num <= 14):
            position_row_char = str(15 - position_row_num)
            position_col_char = dic_num_to_char[position_col_num]
            position_char = position_col_char + position_row_char
            return position_char
        else:
            raise ImpossiblePositionError("Position over the board")
    except ImpossiblePossitionError as e:
        print(e)
        exit(1)


def insert_move(board, move, color):
    board = board.squeeze()
    try:
        if color is 'black' or color is 0:
            color = 0
        elif color is 'white' or color is 1:
            color = 1
        else:
            raise ImpossibleColorError
    except ImpossibleColorError as e:
        print(e)
        exit(1)
    try:
        if type(move) is type(""):
            row_num, col_num = position_char_to_num(move)
        elif type(move) is type(0):
            row_num, col_num = position_char_to_num(position_num_to_char(move))
        elif type(move) is type(()):
            row_num, col_num = move[0], move[1]
        else:
            raise MoveFormatError("Wrong move type")
    except MoveFormatError as e:
        print(e)
        exit(1)
    if board.ndim is 3:
        board[row_num][col_num][color] = 1
    elif board.ndim is 2:
        # color -1, 1 respectively mean black and white
        if color:
            board[row_num][col_num] = 1
        else:
            board[row_num][col_num] = -1
    return board


def plot_board(board, other_board=None):
    fig = plt.figure()
    if other_board is None:
        board = board.squeeze().copy()
        if board.ndim is 3:
            board = np.rollaxis(board, 2)
            board = board[0] * -1 + board[1]
        plt.imshow(board, cmap='gray', interpolation='none')
        plt.grid(True)
        plt.show()
    else:
        board_l = board.squeeze().copy()
        board_r = other_board.squeeze().copy()
        if board_l.ndim is 3:
            board_l = np.rollaxis(board_l, 2)
            board_l = board_l[0] * -1 + board_l[1]
            board_r = np.rollaxis(board_r, 2)
            board_r = board_r[0] * -1 + board_r[1]
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(board_l, cmap='gray', interpolation='none')
        ax1.grid(True)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(board_r, cmap='gray', interpolation='none')
        ax2.grid(True)
        plt.show()


def valid_move(board, move, color):
    """
    check duplicate error
    """
    board = board.squeeze()
    try:
        if color is 'black' or color is 0:
            color = 0
        elif color is 'white' or color is 1:
            color = 1
        else:
            raise ImpossibleColorError
    except ImpossibleColorError as e:
        print(e)
        exit(1)
    try:
        if type(move) is type(""):
            row_num, col_num = position_char_to_num(move)
        elif type(move) is type(0):
            row_num, col_num = position_char_to_num(position_num_to_char(move))
        elif type(move) is type(()):
            row_num, col_num = move[0], move[1]
        else:
            raise MoveFormatError("Wrong move type")
    except MoveFormatError as e:
        print(e)
        exit(1)
    try:
        if board.ndim is 2:
            # color -1, 1 respectively mean black and white
            if board[row_num][col_num] != 0:
                  raise BoardDuplicateError
        elif board.ndim is 3:
            if board[row_num][col_num][color] != 0:
                  raise BoardDuplicateError
    except BoardDuplicateError as e:
        print(e)
        return False
    return True
