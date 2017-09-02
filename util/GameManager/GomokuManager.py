import numpy as np
import sys
from ..error import ImpossibleColorError
from ..UserInterface.board import position_to_tup, check_not_duplicate_move, check_position_not_over_board


class GomokuManager:

    def __init__(self, state=None): 
        """
        board size is supposed to be 15 x 15
        """
        self.last_move = None
        self.finish = False
        if state is None:
            self._state = np.zeros((15, 15), dtype=np.int8)
        else:
            state = state.squeeze()
            if state.ndim is 3:
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
                raise ImpossibleColorError
        except ImpossibleColorError as e:
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

        directions = [np.array([0,1]), np.array([1,1]), np.array([1,0]), np.array([1,-1])]
        for direction in directions:
            back_pos = target_pos + (-1) * direction
            if check_position_not_over_board(back_pos):
                while (self._state[tuple(back_pos)] == color):
                    back_pos -= direction
                    if not check_position_not_over_board(back_pos):
                        break
            front_pos = back_pos + direction
            cnt_connect = 0
            if check_position_not_over_board(front_pos):
                while (self._state[tuple(front_pos)] == color):
                    front_pos += direction
                    cnt_connect += 1
                    if cnt_connect >= 5:
                        print("The game is over!")
                        self.finish = True
                        return True
                    if not check_position_not_over_board(front_pos):
                        break
        return False

    def player_action(self, next_move):
        """
        update the state
        if invalid move, return false
        """
        if check_not_duplicate_move(self._state, next_move):
            self.last_move = position_to_tup(next_move)
            if self.last_turn() == 'black':
                self._state[self.last_move] = 1
            else:
                self._state[self.last_move] = -1
            self.check_done()
            return True
        return False

    # one_step에서 check_done은 따로!
    # 여기까지 수정함.
    """
    MCTS에서 expand_one_time() 할 때 Leaf노드에서 선택한 다음번 move를 입력받아서
    다음번 돌아온 자신의 차례에서의 state와 게임을 끝까지 simulation했을 때의 결과를 반환한다.

    여기에서 policy들은 MCTS에서의 default policy에 해당한다.
    """
    def next_state_with_montecarlo_result(self, next_move, my_policy, enemy_policy):
        # 올바른 입력만 들어온다고 가정
        # 한 수를 두고나서 게임이 끝났는지 확인한다.
        done = self.one_step(next_move)
        if done:
            if self.turn() == 'black':
                return self.state, -1
            else:
                return self.state, 1

        # 상대방도 한 수를 두고 게임이 끝났는지 확인한다.
        enemy_next_move = enemy_policy.get_next_move(self.state)
        done = self.one_step(enemy_next_move)
        if done:
            if self.turn() == 'black':
                return self.state, -1
            else:
                return self.state, 1

        # 돌아온 차례는 반환하기 위해 저장해둔다.
        # 최종적인 결과는 simulate 함수가 처리한다.
        next_state = self.state
        result = self.simulate(my_policy, enemy_policy)

        return next_state, result

    """
    현재 상태에서 게임을 끝까지 (재귀적으로) 시뮬레이션해서 결과를 반환한다.
    policy에 값을 넣어주면 my_policy와 enemy_policy 모두 policy를 공유한다.
    """
    def simulate(self,policy=None,my_policy=None, enemy_policy=None):
        if policy is not None:
            my_policy = enemy_policy = policy
        next_move = my_policy.get_next_move(self.state)
        done = self.one_step(next_move)
        if done:
            if self.turn() == 'black':
                return self.state, -1
            else:
                return self.state, 1

        return self.simulate(policy=policy,my_policy=enemy_policy,enemy_policy=my_policy)

    """
    현재 게임 보드의 상태를 출력한다
    """
    def show_state(self):
        for row in self.state:
            for item in row:
                if item == 0:
                    print('　',end='')
                elif item == 1:
                    print('○',end='')
                else:
                    print('●',end='')
            print()