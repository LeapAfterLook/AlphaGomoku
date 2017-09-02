import numpy as np
import sys
from error import ImpossiblePositionError, NotYourTurnError, MoveFormatError, BoardDuplicateError


class GomokuManager:

    def __init__(self, state=None):
        """
        board size is supposed to be 15 x 15
        """
        self.last_move = None
        if state is None:
            self._state = np.zeros((15, 15), dtype=np.int8)
        else:
            state = state.squeeze()
            if state.ndim == 3:
                state = np.rollaxis(state, 2)
                state = state[0] * -1 + state[1]
            self._state = state

    @property
    def state(self, ndim=3):
        state = np.zeros((2, 15, 15), dtype=np.int8)
        state[0] = self._state == -1
        state[1] = self._state == 1
        return np.rollaxis(state, 0, 3)

    @state.setter
    def state(self, state):
        state = state.squeeze()
        if state.ndim is 3:
            state = np.rollaxis(state, 2)
            state = state[0] * -1 + state[1]
        self._state = state

    @classmethod
    def char_to_index(c):
        return ord(c) - ord('a')

    @classmethod
    def index_to_char(n):
        return chr(n)

    @classmethod
    def position_str_to_num(cls,position_str):
        try:
            if len(position_str) > 3:
                raise ImpossiblePositionError("Too long position description")
            position_row_index = 15 - int(position_str[1:])
            position_col_index = cls.char_to_index(position_str[0])
            return position_row_index, position_col_index
        except ImpossiblePositionError as e:
            print(e)
            exit(1)

    @classmethod
    def position_to_tuple(cls,position):
        try:
            if type(position) is str:
                row_index, col_index = cls.position_str_to_num(position)
            elif type(position) is int:
                row_index, col_index = position // 15, position % 15
            elif type(position) is tuple:
                row_index, col_index = int(position[0]), int(position[1])
            elif type(position) is np.ndarray:
                row_index, col_index = position[0], position[1]
            else:
                raise MoveFormatError("Wrong move type")
            return row_index, col_index
        except MoveFormatError as e:
            print(e)
            exit(1)

    @classmethod
    def check_position_not_over_board(cls,position):
        row_num, col_num = cls.position_to_tuple(position)
        if (0 <= row_num <= 14) and (0 <= col_num <= 14):
            return True
        else:
            return False

    def last_turn(self):
        """
        return whose turn was the last
        when sum of -1, last turn is black
        """
        sum = self._state.sum()
        try:
            if sum == -1:
                return 'black'
            elif sum == 0:
                return 'white'
            else:
                raise NotYourTurnError
        except NotYourTurnError as e:
            print(e)
            exit(1)

    def check_done(self):
        """
        check if the game is over
        for the sake of simplicity, calculated only around the last move
        right, bottom right, bottom, left bottom
        find the stone that is the same color as the last one
        if so, then navigate as backward as possible
        it counts forwards from there and returns True if it is 5 consecutive
        """
        target_pos = np.array(self.last_move)

        if self.last_turn() == 'black':
            color = -1
        else:
            color = 1

        directions = [np.array([0, 1]), np.array([1, 1]), np.array([1, 0]), np.array([1, -1])]
        for direction in directions:
            back_pos = target_pos + (-1) * direction

            while (self._state[tuple(back_pos)] == color) and self.check_position_not_over_board(back_pos):
                back_pos -= direction

            front_pos = back_pos + direction
            cnt_connect = 0
            while (self._state[tuple(front_pos)] == color) and self.check_position_not_over_board(front_pos):
                front_pos += direction
                cnt_connect += 1
                if cnt_connect >= 5:
                    return True

        return False

    def check_not_duplicate_move(self, move):
        board = self._state.squeeze()
        row_index, col_index = self.position_to_tuple(move)
        if board.ndim is 2:
            if board[row_index][col_index] != 0:
                  raise BoardDuplicateError(row_index,col_index)
        elif board.ndim is 3:
            if board[row_index][col_index][0] != 0:
                  raise BoardDuplicateError(row_index,col_index)
            if board[row_index][col_index][1] != 0:
                  raise BoardDuplicateError(row_index,col_index)

    def one_move(self, next_move):
        """
        update the state
        if invalid move, return false
        """
        self.check_not_duplicate_move(next_move)

        self.last_move = self.position_to_tuple(next_move)
        if self.last_turn() == 'black':
            self._state[self.last_move] = 1
        else:
            self._state[self.last_move] = -1

        return self.check_done()

    def request_move(self,policy):
        blacklist = []
        while True:
            try:
                next_move = policy.get_next_move(state=self._state,blacklist=blacklist)
                return next_move
                break
            except BoardDuplicateError as e:
                blacklist.append(next_move)
                print(e)


    """
    MCTS에서 expand_one_time() 할 때 Leaf노드에서 선택한 다음번 move를 입력받아서
    다음번 돌아온 자신의 차례에서의 state와 게임을 끝까지 simulation했을 때의 결과를 반환한다.

    여기에서 policy들은 MCTS에서의 default policy에 해당한다.
    """

    def next_state_with_montecarlo_result(self, next_move, my_policy, enemy_policy):
        # 올바른 입력만 들어온다고 가정
        # 한 수를 두고나서 게임이 끝났는지 확인한다.
        done = self.one_move(next_move)
        if done:
            if self.last_turn() == 'black':
                return self._state, -1
            else:
                return self._state, 1

        # 상대방도 한 수를 두고 게임이 끝났는지 확인한다.
        enemy_next_move = self.request_move(policy=enemy_policy)

        done = self.one_move(enemy_next_move)
        if done:
            if self.last_turn() == 'black':
                return self._state, -1
            else:
                return self._state, 1

        # 돌아온 차례는 반환하기 위해 저장해둔다.
        # 최종적인 결과는 simulate 함수가 처리한다.
        next_state = self._state
        result = self.simulate(my_policy, enemy_policy)

        return next_state, result

    """
    현재 상태에서 게임을 끝까지 (재귀적으로) 시뮬레이션해서 결과를 반환한다.
    policy에 값을 넣어주면 my_policy와 enemy_policy 모두 policy를 공유한다.
    """

    def simulate(self, policy=None, my_policy=None, enemy_policy=None):
        if policy is not None:
            my_policy = enemy_policy = policy
        next_move = self.request_move(policy=my_policy)
        done = self.one_move(next_move)
        if done:
            if self.turn() == 'black':
                return self._state, -1
            else:
                return self._state, 1

        return self.simulate(policy=policy, my_policy=enemy_policy, enemy_policy=my_policy)

    """
    현재 게임 보드의 상태를 출력한다
    """

    def show_state(self):
        print('   ',end='')
        for col_num in range(15):
            print(' {col_num} '.format(col_num=col_num),end='')
        print()

        row_num = 0
        for row in self._state:
            print('{row_num:<3}'.format(row_num=row_num),end='')
            row_num += 1
            for item in row:
                if item == 0:
                    print('   ', end='')
                elif item == 1:
                    print(' {}'.format('○'), end='')
                else:
                    print(' {}'.format('●'), end='')
            print()

    def test(self):
        while True:
            while True:
                try:
                    next_move = tuple(input('next move?').split(' '))
                    self.one_move(next_move)
                    break
                except BoardDuplicateError as e:
                    print(e)
            self.show_state()
            if self.check_done():
                print('{winner} win!'.format(winner=self.last_turn()))
                break
