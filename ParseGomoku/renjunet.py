from xml.etree.ElementTree import parse, tostring
import numpy as np
import os
import matplotlib.pyplot as plt

# Parse the database and deal with records
# Download the database: renju.net/downloads/games.php

cur_dir = os.path.abspath(os.path.dirname(__file__))

def parse_database():
    """
    Parse database (xml files) of renju or gomoku games from renju.net and return element tree about xml file
    """
    file = open(os.path.join(cur_dir, "renjunet_v10_20170716.rif"), encoding='utf-8', errors='ignore')
    tree = parse(file)
    file.close()
    database = tree.getroot()
    return database


def moves_validation_test(moves):
    for move in moves:
        # if move is too short, it means that the move has wrong format
        if len(move) > 3:
            print("MoveFormatError")
            exit(1)

        row_num = int(move[1:])
        col_num = move[0]

        # if the move is not in 15 x 15 board, error occur
        assert (1 <= row_num <= 15)
        assert ('a' <= col_num <= 'o')


def plot_board(board_images, other_board_images=None):
    fig = plt.figure()
    if type(other_board_images) != type(None):
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(board_images * np.full((15, 15), -1), cmap='gray', interpolation='none')
        ax1.grid(True)
        ax2.imshow(other_board_images * np.full((15, 15), -1), cmap='gray', interpolation='none')
        ax2.grid(True)
        plt.show()
    else:
        plt.imshow(board_images * np.full((15, 15), -1), cmap='gray', interpolation='none')
        plt.grid(True)
        plt.show()

# game which has too short sequence (moves) will be eliminated
limit_short_moves = 3

# index dictionary
dic_char_to_num = {}
dic_num_to_char = {}
char_list = ' '.join("abcdefghijklmno").split()
for i in range(0, 15):
    dic_char_to_num[char_list[i]] = i
    dic_num_to_char[i] = char_list[i]


class Record:

    def __init__(self):
        pass

    @classmethod
    def get_len_board_images(cls):
        with open(cur_dir + "/meta_record") as meta_record:
            len_board_images = int(meta_record.readline())

        return len_board_images

    @classmethod
    def make_new_record(cls):
        input_record = open(os.path.join(cur_dir, "input_record"), "wb")
        output_record = open(os.path.join(cur_dir, "output_record"), "wb")
        meta_record = open(os.path.join(cur_dir, "meta_record"), "w")

        database = parse_database()

        cnt_games = 0
        len_board_images = 0

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
            if len(array_moves) <= limit_short_moves:
                continue

            # data validation test
            moves_validation_test(array_moves)

            # make board image for moves
            input_board_images = np.zeros((0, 15, 15), dtype=np.int8)
            output_board_labels = np.zeros((0, 2, 15), dtype=np.bool)

            output_board_image = np.zeros((15, 15), dtype=np.int8)

            # black: 1, white: -1, none: 0
            cnt_moves = 0
            for move in array_moves:
                row_num = 15 - int(move[1:])
                col_num = dic_char_to_num[move[0]]

                # initialize input board image to previous board image
                input_board_image = output_board_image.copy()

                # check duplicate error
                if input_board_image[row_num][col_num] != 0:
                    print("BoardDuplicateError")
                    exit(1)

                # make output board image include next move
                output_board_label = np.zeros((2, 15), dtype=np.int8)
                if cnt_moves % 2 == 0:
                    output_board_image[row_num][col_num] = 1
                    output_board_label[0][row_num] = 1
                    output_board_label[1][col_num] = 1
                else:
                    output_board_image[row_num][col_num] = -1
                    output_board_label[0][row_num] = -1
                    output_board_label[1][col_num] = -1

                input_board_images = np.vstack((input_board_images, input_board_image.reshape(1, 15, 15)))
                output_board_labels = np.vstack((output_board_labels, output_board_label.reshape(1, 2, 15)))

                cnt_moves += 1

            input_record.write(input_board_images.tobytes())
            output_record.write(output_board_labels.tobytes())
            len_board_images += len(input_board_images)

            cnt_games += 1

            # check
            # if cnt_games == 1:
            #     print(output_board_labels[64])
            #     plot_board(input_board_images[64])

        meta_record.write(str(len_board_images))
        input_record.close()
        output_record.close()
        meta_record.close()

        return

    @classmethod
    def load_input_records(cls, count=None):
        if count:
            len_board_images = count
        else:
            len_board_images = cls.get_len_board_images()
        input_board_images = np.zeros((len_board_images, 15, 15))
        with open(os.path.join(cur_dir, "input_record"), "rb") as input_record:
            for k in range(len_board_images):
                input_board_images[k] = np.frombuffer(input_record.read(225), dtype=np.int8).reshape(15, 15)

        print("shape of input_board_images:", input_board_images.shape)
        return input_board_images

    @classmethod
    def load_output_records(cls, count=None, encoding_option='vector'):
        if count:
            len_board_images = count
        else:
            len_board_images = cls.get_len_board_images()
        if encoding_option == 'vector':
            output_board_labels = np.zeros((len_board_images, 2, 15))
            with open(os.path.join(cur_dir, "output_record"), "rb") as output_record:
                for k in range(len_board_images):
                    output_board_labels[k] = np.frombuffer(output_record.read(30), dtype=np.int8).reshape(2, 15)

        elif encoding_option == 'sequence':
            output_board_labels = np.zeros((len_board_images, 225))
            with open(os.path.join(cur_dir, "output_record"), "rb") as output_record:
                for k in range(len_board_images):
                    vector = np.frombuffer(output_record.read(30), dtype=np.int8).reshape(2, 15)
                    output_board_labels[k][vector[0].argmax() * 15 + vector[1].argmax()] = 1

        print("shape of output_board_labels:", output_board_labels.shape)
        return output_board_labels

# make record and test
if __name__ == '__main__':
    # Record.make_new_record()
    x_board_images = Record.load_input_records(500)
    y_board_labels = Record.load_output_records(500, encoding_option='vector')
    print(y_board_labels[270])
    y_board_labels = Record.load_output_records(500, encoding_option='sequence')
    print(y_board_labels[270], y_board_labels[270].argmax())
    plot_board(x_board_images[264], x_board_images[265])
