class MoveFormatError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return "MoveFormatError: %s" % self.msg


class ImpossiblePositionError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return "ImpossiblePositionError: %s" % self.msg


class NotYourTurnError(Exception):
    def __str__(self):
        return "NotYourTurn!"


class BoardDuplicateError(Exception):

    def __init__(self,row_index,col_index):
        self.row_index = row_index
        self.col_index = col_index

    def __str__(self):
        return "BoardDuplicateError" + str(self.row_index) + ' ' + str(self.col_index)
