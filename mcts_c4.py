#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
import numpy as np
import pdb
import sys

class COLOR:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


EPS = 1e-4


def color_board(i, x, y, highlist_pos, color):
    if (x, y) in highlist_pos:
        if color is None:
            s = COLOR.UNDERLINE + str(int(i)) + COLOR.END
        else:
            s = color + str(int(i)) + COLOR.END

    else:
        s = str(int(i))
    if i == 0:
        s = '.'
    elif i == 1:
        s = COLOR.RED + s + COLOR.END
    else:
        s = COLOR.YELLOW + s + COLOR.END
    return s


def display_board(state, pos, color):
    assert type(pos) is set
    s = ''
    for r in range(state.board.shape[0]):
        s += ' '.join([color_board(i, r, idx, pos, color) for idx, i in enumerate(state.board[r, :])]) + '\n'
    return s


class SearchTree(object):
    def __init__(self, game):
        self.game = game
        self.iters = 1000
        self.nodes_seen = {}
        self.gamma = 1.0

    def search(self, root_node):
        self.nodes_seen[str(root_node.state)] = root_node
        for _ in range(self.iters):
            if _ % 100 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            # print('************************ search iter' + str(_) + '*************************')
            search_sequence = []
            node, search_sequence = self.selection(root_node)
            if not node.state.is_terminal:
                node = self.expansion(node)
                search_sequence.append(node)
            reward, winner, winner_pos = self.rollout(node)
            self.backup(node, reward, winner, search_sequence)
            # print('********************** end search iter' + str(_) + '*******************')
        print('')
        # self.display_child_values(root_node, 0.0)
        # print('diplay completed!!!!!!!!!!')
        best_action, best_node = self.select_child(root_node, 0.0)
        return best_action

    def selection(self, node):
        search_sequence = [node]
        while node.completed_expansion:
            assert not node.state.is_terminal
            action, node = self.select_child(node, 0.5)
            search_sequence.append(node)
        return node, search_sequence

    def expansion(self, node):
        assert not node.completed_expansion
        assert not node.state.is_terminal
        child_node = self.expand_child(node)
        return child_node

    def rollout(self, node):
        state = node.state
        steps = 0
        while not state.is_terminal:
            pa = self.game.possible_actions(state)
            selected_action = pa[np.random.choice(len(pa))]
            state = self.game.next_state(state, selected_action)
            steps += 1
            if steps > game.max_steps:
                print('exceeded max steps')
                pdb.set_trace()
        return self.gamma ** steps, state.winner, state.winner_pos

    def backup(self, node, reward, winner, search_sequence):
        # print('------------------backup-----------------')
        for node in search_sequence:
            node.rewards[winner] += reward
            node.visits += 1
        # print('-----------------------------------------')
        return True

    def display_child_values(self, node, exp_param):
        combined, values, ucb, actions, rewards, reward_sum, visits = self.child_scores(node, exp_param)
        f = [(a[1], a[0], c, v, u, w, sw, vs) for a, c, v, u, w, sw, vs in zip(actions, combined, values, ucb, rewards, reward_sum, visits)]
        s = '\n'.join(['combined %0.2f' % c +
                       ' value: %0.2f' % v +
                       ' ucb: %0.2f' % u +
                       ' position:' + str(a0) + ',' + str(a1) +
                       ' rewards:' + w + ' reward_sum:' + sw +
                       ' visits:' + str(vs) for
                       a1, a0, c, v, u, w, sw, vs in sorted(f)])
        print(s)
        return s

    def child_scores(self, node, exp_param):
        n_visits = float(node.visits)
        cn_player = 3 - node.state.player  # make selection from current node to child node
        values = np.zeros(len(node.expanded_actions))
        ucb = np.zeros(len(node.expanded_actions))
        actions = [None] * len(node.expanded_actions)
        rewards = []
        reward_sum = []
        visits = []
        for idx, (a, cn) in enumerate(node.expanded_actions.items()):
            cn_visits = sum(cn.rewards)
            cn_value = (float(cn.rewards[cn_player]) / float(cn_visits))
            cn_ucb = np.sqrt(np.log2(n_visits) / cn_visits)
            assert not np.isnan(cn_value)
            assert not np.isnan(cn_ucb)
            values[idx] = cn_value
            ucb[idx] = cn_ucb
            actions[idx] = a
            rewards.append(' '.join(['%.2f' % w for w in cn.rewards]))
            reward_sum.append('%.2f' % sum(cn.rewards))
            visits.append(cn.visits)
        combined = (1.0 - exp_param) * values + exp_param * ucb
        return combined, values, ucb, actions, rewards, reward_sum, visits

    def select_child(self, node, exp_param):
        combined, values, ucb, actions, rewards, reward_sum, visits = self.child_scores(node, exp_param)
        max_idx = np.random.choice(np.flatnonzero(combined == combined.max()))
        best_action = actions[max_idx]
        return best_action, node.expanded_actions[best_action]

    def expand_child(self, node):
        assert len(node.unexpanded_actions) > 0
        action, _ = node.unexpanded_actions.popitem()
        new_state = self.game.next_state(node.state, action)
        new_rewards = [EPS, EPS, EPS]
        new_state_unexpanded_actions = {a: None for a in self.game.possible_actions(new_state)}
        new_state_expanded_actions = {}
        if str(new_state) in self.nodes_seen:
            expanded_node = self.nodes_seen[str(new_state)]
        else:
            expanded_node = SearchNode(new_state, new_state_unexpanded_actions, new_state_expanded_actions, new_rewards)
            self.nodes_seen[str(new_state)] = expanded_node

        node.update_expansion(action, expanded_node)
        return expanded_node


class SearchNode(object):
    def __init__(self, state, unexpanded_actions, expanded_actions, rewards):
        self.state = state
        self.unexpanded_actions = unexpanded_actions
        self.expanded_actions = expanded_actions
        self.rewards = rewards
        self.visits = 0
        self.completed_expansion = False

    def update_expansion(self, action, child_node):
        assert action not in self.expanded_actions
        self.expanded_actions[action] = child_node
        self.completed_expansion = len(self.unexpanded_actions) == 0
        return True

    def __str__(self,):
        s = str(self.state) + '\n'
        s += 'visits:' + str(self.visits) + '\n'
        s += 'rewards:' + ' '.join(['%.2f' % i for i in self.rewards]) + '\n'
        s += 'u:' + str(len(self.unexpanded_actions)) + ' e:' + str(len(self.expanded_actions)) + '\n'
        s += 'exp complete:' + str(self.completed_expansion) + '\n'
        return s


class Game(object):
    left = [(0, 0), (0, -1), (0, -2), (0, -3)]
    right = [(0, 0), (0, 1), (0, 2), (0, 3)]
    down = [(0, 0), (1, 0), (2, 0), (3, 0)]
    leftdown = [(0, 0), (1, -1), (2, -2), (3, -3)]
    rightdown = [(0, 0), (1, 1), (2, 2), (3, 3)]

    def __init__(self, size):
        self.size = size
        self.max_steps = size[0] * size[1]

    def start(self,):
        b = np.zeros(self.size, dtype=int)
        player = 2
        return State(player, None, b)

    def possible_actions(self, state):
        a = []
        for c in range(state.board.shape[1]):
            for r in reversed(range(state.board.shape[0])):
                if state.board[r, c] == 0:
                    a.append((r, c))
                    break
        return a

    def next_state(self, state, action):
        new_board = state.board.copy()
        new_player = 3 - state.player
        new_board[action[0], action[1]] = new_player
        new_state = State(new_player, action, new_board)
        winner, winner_pos = self.check(new_state)
        pa = self.possible_actions(new_state)
        is_terminal = winner != 0 or len(pa) == 0
        new_state.set_is_terminal(is_terminal, winner, winner_pos)
        return new_state

    def __left_shift(self, state):
        r, c = state.position
        i = 0
        while 0 <= c + i and state.board[r, c + i] == state.board[r, c]:
            i -= 1
        i += 1
        return r, c + i

    def __lefttop_shift(self, state):
        r, c = state.position
        i = 0
        while 0 <= c + i and 0 <= r + i and state.board[r + i, c + i] == state.board[r, c]:
            i -= 1
        i += 1
        return r + i, c + i

    def __righttop_shift(self, state):
        r, c = state.position
        i = 0
        j = 0
        while c + i <= 6 and 0 <= r + j and state.board[r + j, c + i] == state.board[r, c]:
            i += 1
            j -= 1
        j += 1
        i -= 1
        return r + j, c + i

    def check(self, state):
        for name, shift, vec in zip(['left', 'down', 'lefttop', 'righttop'],
                                    [self.__left_shift, None, self.__lefttop_shift, self.__righttop_shift],
                                    [Game.right, Game.down, Game.rightdown, Game.leftdown]):
            if shift is not None:
                r, c = shift(state)
            else:
                r, c = state.position

            cl = [(i[0] + r, i[1] + c) for i in vec if (0 <= i[0] + r < self.size[0] and 0 <= i[1] + c < self.size[1])]
            if len(cl) == 4:
                _cl = list(zip(*cl))
                cc = state.board[_cl].tolist()
                if cc == [state.player] * 4:
                    return state.player, set(cl)
        return 0, set([])


class State(object):
    def __init__(self, player, position, board):
        self.board = board
        self.position = position
        self.player = player  # player whos move lead to this state, i.e. the last player
        self.is_terminal = False
        self.winner = None
        self.winner_pos = None

    def set_is_terminal(self, is_terminal, winner, winner_pos):
        self.is_terminal = is_terminal
        self.winner = winner
        self.winner_pos = winner_pos

    def __str__(self,):
        s = 'player:' + str(self.player) + '\n'
        s += 'is_terminal:' + str(self.is_terminal) + '\n'
        s += 'board:' + '\n'
        for r in range(self.board.shape[0]):
            s += ''.join([str(int(i)) for idx, i in enumerate(self.board[r, :])]) + '\n'
        return s.strip()


def mcts_search(state, game):
    search_tree = SearchTree(game)
    unexpanded_actions = {a: None for a in game.possible_actions(state)}
    expanded_actions = {}
    root_node = SearchNode(state, unexpanded_actions, expanded_actions, [EPS, EPS, EPS])
    best_action = search_tree.search(root_node)
    return best_action


if __name__ == '__main__':
    game = Game((6, 7))
    state = game.start()
    winner = 0
    while len(game.possible_actions(state)) > 0 and winner == 0:
        print(display_board(state, set([state.position]), None))
        print('player' + str(3 - state.player) + ':')
        if 3 - state.player == 1:
            action_map = {i[1]: i for i in game.possible_actions(state)}
            i = -1
            while i not in action_map:
                i = input('select action: ' + ','.join([str(k) for k, v in action_map.items()]) + ':')
            selected_action = action_map[int(i)]
        else:
            selected_action = mcts_search(state, game)
        state = game.next_state(state, selected_action)
        winner, winner_pos = game.check(state)
    print(display_board(state, set(winner_pos), COLOR.GREEN))
    print('game over!')
