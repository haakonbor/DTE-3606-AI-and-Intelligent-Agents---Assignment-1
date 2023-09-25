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

        # State info
        self.terminal_states = [[0, 8], [5, 9], [8, 6]]
        self.blocked_states = [[1, 2], [2, 2], [3, 2], [4, 2], [3, 6], [4, 6], [5, 6], [6, 6], [5, 7], [6, 7]]

        # Dimensions
        self.state_space = 100
        self.action_space = action_space

        # R-matrix and Q-table
        self.r_table = np.zeros((10, 10))
        self.q_table = np.zeros((self.state_space, self.action_space))

        # Terminal state 1: -500
        self.r_table[0][8] -= 500

        # Terminal state 2: +1000
        self.r_table[5][9] += 1000

        # Terminal state 3: -100
        self.r_table[8][6] -= 100
        self.rewards_all_episodes = []
        self.steps_all_episodes = []

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

    def learn(self):
        for episode in range(self.episodes):
            # Start in a random state in either row "A" or "I", and before column 4.
            state = [random.choice([0, 9]), 0]

            rewards_curr_episode = []
            steps_curr_episode = 0

            actions_made = []
            states_visited = []

            while not self.in_terminal_state(state):
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
                reward = self.r_table[new_state[0], new_state[1]] + movement_cost
                rewards_curr_episode.append(reward)
                steps_curr_episode += movement_cost / -5

                prev_q = self.q_table[q_index, action]

                # Chapter 6.5 in "Reinforcement Learning - An Introduction, 2nd edition"
                # by Richard S. Sutton and Andrew G. Barto
                q = prev_q + self.learning_rate * \
                    (reward + self.discount_factor * max(self.q_table[new_q_index]) - prev_q)
                self.q_table[q_index, action] = q
                state = new_state

            sum_rewards_curr_episode = np.sum(rewards_curr_episode)
            self.rewards_all_episodes.append(sum_rewards_curr_episode)
            self.steps_all_episodes.append(steps_curr_episode)

    def stats(self):
        # print("----- UPDATED Q-TABLE ------\n")
        # print(q_table)

        print("----- MAX Q ------\n")
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
        #
        # rewards_per_10_ep = np.split(np.array(self.rewards_all_episodes), int(self.episodes / 10))
        # print("----- AVG REWARD PER 10 EPISODE ------\n")
        # count = 10
        # for rewards in rewards_per_10_ep:
        #     print(count, ": ", str(sum(rewards)/10))
        #     count += 10
        #
        # steps_per_10_ep = np.split(np.array(self.steps_all_episodes), int(self.episodes / 10))
        # print("----- AVG STEPS PER 10 EPISODE ------\n")
        # count = 10
        # for steps in steps_per_10_ep:
        #     print(count, ": ", str(sum(steps) / 10))
        #     count += 10

        x = range(0, self.episodes)
        y = self.rewards_all_episodes
        avg_range = 100
        avg_rewards = []
        for i in range(len(y) - avg_range + 1):
            avg_rewards.append(np.mean(y[i: i + avg_range]))
        for i in range(avg_range - 1):
            avg_rewards.insert(0, np.nan)

        fig, ax = plt.subplots()
        ax.plot(x, avg_rewards)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Score')
        ax.set_title(f'Running average score (window = {avg_range})')
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
        self.hideouts_prob = [0.25, 0.75]
        self.thief_pos = self.hideouts[0]
        self.r_table[self.thief_pos[0], self.thief_pos[1]] = 1000
        self.terminal_states.append(self.thief_pos)

    def transition(self, state, action):
        new_state = super().transition(state, action)

        for hideout in self.hideouts:
            self.r_table[hideout] = 0

        self.thief_pos = self.hideouts[np.random.choice(len(self.hideouts), p=self.hideouts_prob)]
        self.r_table[self.thief_pos[0], self.thief_pos[1]] = 1000
        self.terminal_states[0] = self.thief_pos

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

    else:
        print("Subtask not implemented yet or invalid input")