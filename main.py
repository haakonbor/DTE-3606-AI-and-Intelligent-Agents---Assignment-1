import numpy as np
# import time
import matplotlib.pyplot as plt
from enum import IntEnum

# np.random.seed(int(time.time()))
np.random.seed(0)  # For reproducing results
np.set_printoptions(precision=1)

subtask = None


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
        self.max_steps = 1000

        # Number of runs
        self.runs = 3

        # State info
        self.terminal_states = [[5, 9], [0, 8], [8, 6]]
        self.blocked_states = [[1, 2], [2, 2], [3, 2], [4, 2], [3, 6], [4, 6], [5, 6], [6, 6], [5, 7], [6, 7]]

        # Dimensions
        self.state_space = 100
        self.action_space = action_space

        # Visits to each state-action pair
        self.state_action_visits = np.zeros((self.state_space, self.action_space))

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

        # Statistics
        self.score = np.zeros((self.runs, self.episodes))
        self.steps = np.zeros((self.runs, self.episodes))
        self.abs_td_error = np.zeros((self.runs, self.episodes))

        self.max_q = np.zeros((10, 10))

        self.avg_range = 50

        self.fig, self.axs = plt.subplots(2, 2)
        self.x = range(0, self.episodes)

    def policy(self, q_index):
        random_num = np.random.uniform(0, 1)

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

        # Blocked actions lead to agent waiting instead
        if new_state[0] < 0 or new_state[0] > 9 or new_state[1] < 0 or new_state[1] > 9 \
                or new_state in self.blocked_states:
            new_state = state.copy()

        return new_state

    @staticmethod
    def get_q_index(state):
        return state[0] * 10 + state[1]

    def get_learning_rate(self, state=None, action=None):
        return self.learning_rate

    def in_terminal_state(self, state):
        return state in self.terminal_states

    def reset_q_table(self):
        self.q_table.fill(0)
        self.state_action_visits.fill(0)

    def learn(self):
        for run in range(self.runs):
            self.reset_q_table()
            for episode in range(self.episodes):
                # Starting state
                state = self.terminal_states[0]
                while state in self.terminal_states or state in self.blocked_states:
                    state = [np.random.choice(9), np.random.choice(9)]

                # For debug purposes
                actions_made = []
                states_visited = []

                while not self.in_terminal_state(state) and self.steps[run][episode] < self.max_steps:
                    q_index = self.get_q_index(state)

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
                    reward = self.r_table[new_state[0], new_state[1]]
                    reward_prob = self.reward_prob[new_state[0], new_state[1]]
                    self.score[run][episode] += reward + movement_cost
                    self.steps[run][episode] += movement_cost / -5

                    prev_q = self.q_table[q_index, action]

                    # Q-learning algorithm
                    td_error = (reward_prob * reward + movement_cost + self.discount_factor
                                * max(self.q_table[new_q_index]) - prev_q)
                    self.abs_td_error[run][episode] += abs(td_error)
                    learning_rate = self.get_learning_rate(state, action)
                    q = prev_q + learning_rate * td_error
                    self.q_table[q_index, action] = q

                    self.state_action_visits[q_index, action] += 1
                    state = new_state

    def plot_rewards(self):
        for run in range(self.runs):
            y = self.score[run]
            avg_rewards = []
            for i in range(len(y) - self.avg_range + 1):
                avg_rewards.append(np.mean(y[i: i + self.avg_range]))
            for i in range(self.avg_range - 1):
                avg_rewards.insert(0, np.nan)
            self.axs[0][0].plot(self.x, avg_rewards)

        self.axs[0][0].set_xlabel('Episodes')
        self.axs[0][0].set_ylabel('Score')
        self.axs[0][0].set_title(f'Rolling average score (window = {self.avg_range})')

    def plot_steps(self):
        for run in range(self.runs):
            y = self.steps[run]
            avg_steps = []
            for i in range(len(y) - self.avg_range + 1):
                avg_steps.append(np.mean(y[i: i + self.avg_range]))
            for i in range(self.avg_range - 1):
                avg_steps.insert(0, np.nan)
            self.axs[0][1].plot(self.x, avg_steps)
        self.axs[0][1].set_xlabel('Episodes')
        self.axs[0][1].set_ylabel('Steps')
        self.axs[0][1].set_title(f'Rolling average steps (window = {self.avg_range})')

    def plot_abs_td_errors(self):
        for run in range(self.runs):
            y = self.abs_td_error[run]
            avg_error = []
            for i in range(len(y) - self.avg_range + 1):
                avg_error.append(np.mean(y[i: i + self.avg_range]))
            for i in range(self.avg_range - 1):
                avg_error.insert(0, np.nan)
            self.axs[1][0].plot(self.x, avg_error)
        self.axs[1][0].set_xlabel('Episodes')
        self.axs[1][0].set_ylabel('Error')
        self.axs[1][0].set_title(f'Rolling average absolute TD error (window = {self.avg_range})')

    def plot_max_q_heatmap(self):
        max_q = np.zeros((2, 10, 10))
        row_index = 0
        col_index = 0
        for q_values in self.q_table:
            max_q[0][row_index][col_index] = max(q_values)
            max_q[1][row_index][col_index] = np.argmax(q_values)
            col_index += 1
            if col_index >= 10:
                col_index = 0
                row_index += 1

        self.axs[1][1].imshow(max_q[0])
        self.axs[1][1].set_xticks(np.arange(10), labels=range(1, 11))
        self.axs[1][1].set_yticks(np.arange(10), labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

        for i in range(10):
            for j in range(10):
                action = max_q[1][i][j]
                text = ""
                if [i, j] not in self.blocked_states and (self.action_space > 4 or [i, j] not in self.terminal_states):
                    if action == Action.LEFT:
                        text = "←"
                    elif action == Action.UP:
                        text = "↑"
                    elif action == Action.RIGHT:
                        text = "→"
                    elif action == Action.DOWN:
                        text = "↓"
                    else:
                        text = "-"
                self.axs[1][1].text(j, i, text, ha="center", va="center", color="w")

        self.axs[1][1].set_title(f'Heatmap of max Q-values in each state')

    def stats(self):
        self.plot_rewards()
        self.plot_steps()
        self.plot_abs_td_errors()
        self.plot_max_q_heatmap()

        self.fig.set_size_inches(12, 8)
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle(f'Task {subtask}) with parameters:\n learning rate: {self.learning_rate}, discount factor: '
                     f'{self.discount_factor}, exploration rate: {self.exploration_rate}')
        plt.show()

        return


class QLearningBountyHunter(QLearning):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.learning_rate = 0.8
        self.discount_factor = 0.8
        self.exploration_rate = 0.2
        self.r_table[5][9] = 0
        self.hideouts = [[5, 1], [6, 8]]
        self.hideouts_prob = [0.35, 0.65]
        self.thief_pos = self.hideouts[0]
        self.terminal_states[0] = self.thief_pos  # Replace loot terminal state with thief position
        self.r_table[self.thief_pos[0], self.thief_pos[1]] = 1000
        self.reward_prob[self.hideouts[0][0], self.hideouts[0][1]] = self.hideouts_prob[0]
        self.reward_prob[self.hideouts[1][0], self.hideouts[1][1]] = self.hideouts_prob[1]

    def get_learning_rate(self, state=None, action=None):
        return 1 / (1 + self.state_action_visits[self.get_q_index(state), action])

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
        self.steps = np.zeros((self.runs, self.episodes, 2))

    def learn(self):
        for run in range(self.runs):
            self.reset_q_table()
            for episode in range(self.episodes):
                states = [self.terminal_states[0], self.terminal_states[0]]

                while states[0] in self.terminal_states or states[0] in self.blocked_states:
                    states[0] = [np.random.choice(9), np.random.choice(9)]

                while states[1] in self.terminal_states or states[1] in self.blocked_states or states[1] == states[0]:
                    states[1] = [np.random.choice(9), np.random.choice(9)]

                actions_made = [[], []]
                states_visited = [[], []]

                while (not self.in_terminal_state(states[0]) and not self.in_terminal_state(states[1])
                       and not self.steps[run][episode][0] > self.max_steps):
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
                    reward = self.r_table[new_states[0][0], new_states[0][1]]
                    reward_prob = self.reward_prob[new_states[0][0], new_states[0][1]]
                    self.score[run][episode] += reward + movement_cost
                    self.steps[run][episode][0] += movement_cost / -5

                    prev_q = self.q_table[q_indexes[0], actions[0]]
                    td_error = (reward_prob * reward + movement_cost + self.discount_factor
                                * max(self.q_table[new_q_indexes[0]]) - prev_q)
                    self.abs_td_error[run][episode] += abs(td_error)
                    q = prev_q + self.get_learning_rate(states[0], actions[0]) * td_error
                    self.q_table[q_indexes[0], actions[0]] = q

                    self.state_action_visits[q_indexes[0], actions[0]] += 1
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
                    reward = self.r_table[new_states[1][0], new_states[1][1]]
                    reward_prob = self.reward_prob[new_states[1][0], new_states[1][1]]
                    self.score[run][episode] += reward + movement_cost
                    self.steps[run][episode][1] += movement_cost / -5

                    prev_q = self.q_table[q_indexes[1], actions[1]]
                    td_error = (reward_prob * reward + movement_cost + self.discount_factor
                                * max(self.q_table[new_q_indexes[1]]) - prev_q)
                    self.abs_td_error[run][episode] += abs(td_error)
                    q = prev_q + self.get_learning_rate(states[1], actions[1]) * td_error
                    self.q_table[q_indexes[1], actions[1]] = q

                    self.state_action_visits[q_indexes[1], actions[1]] += 1
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
