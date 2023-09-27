import random
import numpy as np
import time
import matplotlib.pyplot as plt
from enum import IntEnum

np.random.seed(int(time.time()))
np.set_printoptions(precision=1)


class Action(IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    WAIT = 4


class QLearning:
    def __init__(self, action_space):
        # Hyper parameters
        self.learning_rate = 0.8
        self.discount_factor = 0.8
        self.exploration_rate = 0.1
        self.episodes = 1000
        self.max_steps = 500

        # State info
        self.terminal_states = [[0, 8], [5, 9], [8, 6]]
        self.blocked_states = [[1, 2], [2, 2], [3, 2], [4, 2], [3, 6], [4, 6], [5, 6], [6, 6], [5, 7], [6, 7]]

        # Dimensions
        self.state_space = 100
        self.action_space = action_space

        # R-matrix and Q-table
        self.r_table = np.zeros((10, 10))
        self.q_table = np.zeros((self.state_space, self.action_space))

        # Probability of gaining reward
        self.reward_prob = np.zeros((10, 10))

        # Terminal state 1: -500
        self.r_table[0][8] -= 500
        self.reward_prob[0][8] = 1

        # Terminal state 2: +1000
        self.r_table[5][9] += 1000
        self.reward_prob[5][9] = 1

        # Terminal state 3: -100
        self.r_table[8][6] -= 100
        self.reward_prob[8][6] = 1

        self.rewards = np.zeros(self.episodes)
        self.steps = np.zeros(self.episodes)
        self.abs_td_error = np.zeros(self.episodes)

    def policy(self, q_index):
        random_num = random.uniform(0, 1)

        # Exploitation
        if random_num > self.exploration_rate:
            action = np.argmax(self.q_table[q_index, :])

        # Exploration
        else:
            action = np.random.choice(len(self.q_table[q_index]))

        return action

    def transition(self, state, action):
        new_state = []

        if action == Action.LEFT:
            new_state = [state[0], state[1] - 1]
        elif action == Action.UP:
            new_state = [state[0] - 1, state[1]]
        elif action == Action.RIGHT:
            new_state = [state[0], state[1] + 1]
        elif action == Action.DOWN:
            new_state = [state[0] + 1, state[1]]
        elif action == Action.WAIT:
            new_state = state.copy()

        if new_state[0] < 0 or new_state[0] > 9 or new_state[1] < 0 or new_state[1] > 9 \
                or new_state in self.blocked_states:
            new_state = state.copy()

        return new_state

    def in_terminal_state(self, state):
        return state in self.terminal_states

    def reward_func(self, new_state, movement_cost):
        return self.r_table[new_state[0], new_state[1]] + movement_cost

    def learn(self):
        for episode in range(self.episodes):
            state = [0, 0]

            actions_made = []
            states_visited = []

            while not self.in_terminal_state(state) and self.steps[episode] < self.max_steps:
                q_index = state[0] * 10 + state[1]

                # Action chosen using epsilon-greedy policy
                action = self.policy(q_index)

                new_state = self.transition(state, action)
                new_q_index = new_state[0] * 10 + new_state[1]

                # Hit a wall (or waiting in the case of the bounty hunter)
                if new_state == state and self.action_space == 4:
                    continue

                actions_made.append(action)
                states_visited.append(state)

                movement_cost = 0 if action == Action.WAIT else -5
                reward = self.reward_func(new_state, movement_cost)
                reward_prob = self.reward_prob[new_state[0], new_state[1]]
                self.rewards[episode] += reward
                self.steps[episode] += movement_cost / -5

                prev_q = self.q_table[q_index, action]

                # Q-learning algorithm
                td_error = reward_prob * reward + self.discount_factor * max(self.q_table[new_q_index]) - prev_q
                self.abs_td_error[episode] += abs(td_error)
                q = prev_q + self.learning_rate * td_error
                self.q_table[q_index, action] = q
                state = new_state

    def stats(self):
        print("----- MAX Q ------")
        max_q = np.zeros((10, 10))
        row_index = 0
        col_index = 0
        for q_values in self.q_table:
            max_q[row_index][col_index] = max(q_values)
            col_index += 1
            if col_index >= 10:
                col_index = 0
                row_index += 1
        print(max_q)

        fig, axs = plt.subplots(3)
        x = range(0, self.episodes)

        y = self.rewards
        avg_range = 10
        avg_rewards = []
        for i in range(len(y) - avg_range + 1):
            avg_rewards.append(np.mean(y[i: i + avg_range]))
        for i in range(avg_range - 1):
            avg_rewards.insert(0, np.nan)
        axs[0].plot(x, avg_rewards)
        axs[0].set_xlabel('Episodes')
        axs[0].set_ylabel('Score')
        axs[0].set_title(f'Running average score (window = {avg_range})')

        y = self.steps
        avg_range = 10
        avg_steps = []
        for i in range(len(y) - avg_range + 1):
            avg_steps.append(np.mean(y[i: i + avg_range]))
        for i in range(avg_range - 1):
            avg_steps.insert(0, np.nan)
        axs[1].plot(x, avg_steps)
        axs[1].set_xlabel('Episodes')
        axs[1].set_ylabel('Steps')
        axs[1].set_title(f'Running average steps (window = {avg_range})')

        y = self.abs_td_error
        avg_range = 10
        avg_error = []
        for i in range(len(y) - avg_range + 1):
            avg_error.append(np.mean(y[i: i + avg_range]))
        for i in range(avg_range - 1):
            avg_error.insert(0, np.nan)
        axs[2].plot(x, avg_error)
        axs[2].set_xlabel('Episodes')
        axs[2].set_ylabel('Error')
        axs[2].set_title(f'Running average absolute TD error (window = {avg_range})')

        fig.set_size_inches(8, 8)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return


class QLearningBountyHunter(QLearning):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.learning_rate = 0.8
        self.discount_factor = 0.8
        self.exploration_rate = 0.1
        for terminal_state in self.terminal_states:
            self.r_table[terminal_state] = 0
        self.hideouts = [[5, 1], [6, 8]]
        self.hideouts_prob = [0.35, 0.65]
        for i in range(len(self.hideouts)):
            self.r_table[self.hideouts[i][0], self.hideouts[i][1]] = 1000
            self.reward_prob[self.hideouts[i][0], self.hideouts[i][1]] = self.hideouts_prob[i]
        self.thief_pos = self.hideouts[0]
        self.terminal_states.append(self.thief_pos)

    def move_thief(self):
        for hideout in self.hideouts:
            self.r_table[hideout[0], hideout[1]] = 0
        # Thief picks to move or not with given probability
        self.thief_pos = self.hideouts[np.random.choice(len(self.hideouts), p=self.hideouts_prob)]
        self.r_table[self.thief_pos[0], self.thief_pos[1]] = 1000
        self.terminal_states[0] = self.thief_pos

    def transition(self, state, action):
        new_state = super().transition(state, action)
        self.move_thief()
        return new_state


class QLearningBountyHunterWithAssistant(QLearningBountyHunter):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.thief_moved = False
        self.steps = np.zeros((1000, 2))

    def learn(self):
        for episode in range(self.episodes):
            states = [[0, 0], [9, 0]]
            actions_made = [[], []]
            states_visited = [[], []]

            while not self.in_terminal_state(states[0]) and not self.in_terminal_state(states[1]):
                self.thief_moved = False
                q_indexes = [states[0][0] * 10 + states[0][1], states[1][0] * 10 + states[1][1]]
                actions = [self.policy(q_indexes[0]), self.policy(q_indexes[1])]
                new_states = [self.transition(states[0], actions[0]), self.transition(states[1], actions[1])]
                new_q_indexes = [new_states[0][0] * 10 + new_states[0][1], new_states[1][0] * 10 + new_states[1][1]]

                """ BOUNTY HUNTER """
                # Bounty hunter is trying to move into state of assistant who is standing still
                if new_states[0] == new_states[1] and actions[1] == Action.WAIT:
                    continue

                actions_made[0].append(actions[0])
                states_visited[0].append(states[0])

                movement_cost = 0 if actions[0] == Action.WAIT else -5
                reward = self.reward_func(new_states[0], movement_cost)
                reward_prob = self.reward_prob[new_states[0][0], new_states[0][1]]
                self.rewards[episode] += reward
                self.steps[episode][0] += movement_cost / -5

                prev_q = self.q_table[q_indexes[0], actions[0]]

                td_error = reward_prob * reward + self.discount_factor * max(self.q_table[new_q_indexes[0]]) - prev_q
                self.abs_td_error[episode] += abs(td_error)
                q = prev_q + self.learning_rate * td_error
                self.q_table[q_indexes[0], actions[0]] = q
                states[0] = new_states[0]

                """ ASSISTANT """
                # Assistant is trying to move into state of the bounty hunter
                if new_states[1] == states[0]:
                    # Make the assistant wait while the bounty hunter is moving into the suggested new state
                    new_states[1] = states[1]
                    actions[1] = Action.WAIT
                    new_q_indexes[1] = new_states[1][0] * 10 + new_states[1][1]

                actions_made[1].append(actions[1])
                states_visited[1].append(states[1])

                movement_cost = 0 if actions[1] == Action.WAIT else -5
                reward = self.reward_func(new_states[1], movement_cost)
                reward_prob = self.reward_prob[new_states[1][0], new_states[1][1]]
                self.rewards[episode] += reward
                self.steps[episode][1] += movement_cost / -5

                prev_q = self.q_table[q_indexes[1], actions[1]]

                td_error = reward_prob * reward + self.discount_factor * max(self.q_table[new_q_indexes[1]]) - prev_q
                self.abs_td_error[episode] += abs(td_error)
                q = prev_q + self.learning_rate * td_error
                self.q_table[q_indexes[1], actions[1]] = q
                states[1] = new_states[1]

    def transition(self, state, action):
        new_state = QLearning.transition(self, state, action)
        if not self.thief_moved:
            self.move_thief()
            self.thief_moved = True
        return new_state


if __name__ == '__main__':
    subtask = input("Choose subtask (a, b, c, ...): ")

    if subtask == "a":
        a = QLearning(4)
        a.learn()
        a.stats()

    elif subtask == "b":
        b = QLearningBountyHunter(5)
        b.learn()
        b.stats()

    elif subtask == "c":
        c = QLearningBountyHunterWithAssistant(5)
        c.learn()
        c.stats()
        pass

    else:
        print("Subtask not implemented yet or invalid input")