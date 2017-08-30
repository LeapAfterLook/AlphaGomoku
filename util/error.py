class MoveFormatError(Exception):
    def __init(self, msg):
        self.msg = msg
    def __str__(self):
        return "MoveFormatError: %s" % self.msg


class ImpossiblePositionError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return "ImpossiblePositionError: %s" % self.msg


class ImpossibleColorError(Exception):
    def __str__(self):
        return "ImpossibleColorError"


class BoardDuplicateError(Exception):
    def __str__(self):
        return "BoardDuplicateError"
