from xml.etree.ElementTree import parse, tostring
import numpy as np
import os

import sys
if not ".." in sys.path:
    sys.path.append("..")
from UserInterface.board import position_char_to_num, insert_move
from error import MoveFormatError, BoardDuplicateError

# Parse the database and deal with records
# Download the database: renju.net/downloads/games.php

cur_dir = os.path.abspath(os.path.dirname(__file__))


def parse_database():
    """
    Parse database (xml files) of renju or gomoku games from renju.net and return element tree about xml file
    """
    with open(os.path.join(cur_dir, "renjunet_v10_20170716.rif"), encoding='utf-8', errors='ignore') as file:
        tree = parse(file)
        database = tree.getroot()
        return database


def moves_validation_test(moves):
    for move in moves:
        try:
            # if move is too long, it means that the move has wrong format
            if len(move) > 3:
                raise MoveFormatError("the move is too long format")
            row_num = int(move[1:])
            col_num = move[0]
            # if the move is not in 15 x 15 board, error occur
            if not (1 <= row_num <= 15):
                raise MoveFormatError("the move is over the board")
            if not ('a' <= col_num <= 'o'):
                raise MoveFormatError("the move is over the board")
        except MoveFormatError as e:
            print(e)
            exit(1)

# game which has too short sequence (moves) will be eliminated
limit_moves = 3


class Record:

    def __init__(self):
        pass

    @classmethod
    def get_len_boards(cls):
        with open(os.path.join(cur_dir, "meta_record")) as meta_record:
            len_boards = int(meta_record.readline())
            return len_boards

    @classmethod
    def make_new_record(cls, record_encoding_option=2):
        """
        record_encoding_option
        1. black: -1, white: 1, none: 0
        2. 0: black, 1: white
        """
        input_record = open(os.path.join(cur_dir, "input_record"), "wb")
        output_record = open(os.path.join(cur_dir, "output_record"), "wb")
        meta_record = open(os.path.join(cur_dir, "meta_record"), "w")

        database = parse_database()

        cnt_games = 0
        len_boards = 0

        for moves in database.getiterator("move"):
            # tag element to bytes string
            bytes_str_moves = tostring(moves)
            # trim tag
            bytes_str_moves = bytes_str_moves[6:-8]
            # bytes string to string
            str_moves = bytes_str_moves.decode("utf-8")
            # split moves string to list of moves
            array_moves = str_moves.split()

            # too short sequence is eliminated
            if len(array_moves) <= limit_moves:
                continue

            # data validation test
            moves_validation_test(array_moves)

            # make board image for moves
            input_boards = np.zeros((0, 15, 15, record_encoding_option), dtype=np.int8)
            output_labels = np.zeros((0, 2, 15), dtype=np.int)
            output_board = np.zeros((15, 15, record_encoding_option), dtype=np.int8)

            cnt_moves = 0  # if cnt_moves is even (when color is 0), next move is black
            for move in array_moves:
                row_num, col_num = position_char_to_num(move)

                # initialize input board image to previous board image
                input_board = output_board.copy()

                color = cnt_moves % 2
                # check duplicate error
                try:
                    if record_encoding_option is 1:
                        if input_board[row_num][col_num][0] != 0:
                            raise BoardDuplicateError
                    elif record_encoding_option is 2:
                        if input_board[row_num][col_num][color] != 0:
                            raise BoardDuplicateError
                except BoardDuplicateError as e:
                    print(e)
                    exit(1)

                # make output board image include next move
                output_board_label = np.zeros((2, 15), dtype=np.int)
                insert_move(output_board, move, color)
                output_board_label[0][row_num] = 1
                output_board_label[1][col_num] = 1

                input_boards = np.vstack((input_boards, np.expand_dims(input_board, 0)))
                output_board_labels = np.vstack((output_board_labels, np.expand_dims(output_board_label, 0)))

                cnt_moves += 1

            input_record.write(input_boards.tobytes())
            output_record.write(output_board_labels.tobytes())
            len_boards += len(input_boards)

            cnt_games += 1

        meta_record.write(str(len_boards))
        input_record.close()
        output_record.close()
        meta_record.close()

    @classmethod
    def load_record_input(cls, count=None, record_encoding_option=2):
        if count:
            len_boards = count
        else:
            len_boards = cls.get_len_boards()
        input_boards = np.zeros((len_boards, 15, 15, record_encoding_option))
        with open(os.path.join(cur_dir, "input_record"), "rb") as input_record:
            for k in range(len_boards):
                input_boards[k] = np.frombuffer(input_record.read(225 * record_encoding_option), dtype=np.int8).reshape(15, 15, record_encoding_option)
        print("shape of input_boards:", input_boards.shape)
        return input_boards

    @classmethod
    def load_record_output(cls, count=None, record_decoding_option='vector'):
        if count:
            len_boards = count
        else:
            len_boards = cls.get_len_boards()
        if encoding_option == 'vector':
            output_board_labels = np.zeros((len_boards, 2, 15))
            with open(os.path.join(cur_dir, "output_record"), "rb") as output_record:
                for k in range(len_boards):
                    output_board_labels[k] = np.frombuffer(output_record.read(30), dtype=np.int).reshape(2, 15)
        elif encoding_option == 'sequence':
            output_board_labels = np.zeros((len_boards, 225))
            with open(os.path.join(cur_dir, "output_record"), "rb") as output_record:
                for k in range(len_boards):
                    vector = np.frombuffer(output_record.read(30), dtype=np.int8).reshape(2, 15)
                    output_board_labels[k][vector[0].argmax() * 15 + vector[1].argmax()] = 1
        print("shape of output_board_labels:", output_board_labels.shape)
        return output_board_labels

# make record and test
if __name__ == '__main__':
    Record.make_new_record()
    x_boards = Record.load_record_input(500)
    y_board_labels = Record.load_record_output(500, encoding_option='vector')
    print(y_board_labels[270])
    y_board_labels = Record.load_record_output(500, encoding_option='sequence')
    print(y_board_labels[270], y_board_labels[270].argmax())
    plot_board(x_board_images[264], x_board_images[265])
