import argparse
import random

from environment import MountainCar
import numpy as np

parser = argparse.ArgumentParser(description="Reinforcement Learning")
parser.add_argument('mode', type=str)
parser.add_argument('weight_out', type=str)
parser.add_argument('returns_out', type=str)
parser.add_argument('episodes', type=int)
parser.add_argument('max_iterations', type=int)
parser.add_argument('epsilon', type=float)
parser.add_argument('gamma', type=float)
parser.add_argument('learning_rate', type=float)


def write_file(output, output_path, method='w'):
    """
    :param output:
    :param output_path:
    :param method: create a new file or append
    :param flag: write number or text
    :return:
    """
    with open(output_path, method) as file:
        for word in output:
            file.write(str(word))
            file.write('\n')


class LinearModel:
    def __init__(self, state_size: int, action_size: int,
                 lr: float, indices: bool):
        """indices is True if indices are used as input for one-hot features.
            Otherwise, use the sparse representation of state as features
        """
        self.lr = lr
        self.indices = indices
        self.weights = np.zeros([state_size, action_size])
        self.bias = 0

    def predict(self, state: dict[int, int]) -> list[float]:
        """
        Given state, makes predictions.
        """
        if self.indices:
            state_array = np.zeros((1, self.weights.shape[0]))[0]
            for key in state.keys():
                state_array[key] = 1
        else:
            state_array = np.array(list(state.values())).reshape(1, -1)[0]
        result = np.dot(state_array, self.weights)+self.bias
        return result.tolist()

    def update(self, state: dict[int, int], action: int, target: float):
        """
        Given state, action, and target, update weights.
        """
        if self.indices:
            state_array = np.zeros((1, self.weights.shape[0]))[0]
            for key in state.keys():
                state_array[key] = 1
        else:
            state_array = np.array(list(state.values())).reshape(1, -1)[0]

        diff = self.predict(state)[action] - target
        self.weights[:, action] = self.weights[:, action] - self.lr * diff * state_array
        self.bias = self.bias - self.lr * diff


class QLearningAgent:
    def __init__(self, env, mode, gamma, lr, epsilon):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        if mode == "tile":
            self.model = LinearModel(env.state_space, env.action_space, self.lr, True)
        if mode == "raw":
            self.model = LinearModel(env.state_space, env.action_space, self.lr, False)

    def get_action(self, state: dict[int, int]) -> int:
        """epsilon-greedy strategy.
        Given state, returns action.
        """
        choice = random.choices(["random", "greedy"], [self.epsilon, 1 - self.epsilon], k=1)[0]
        if choice == "greedy":
            action_result = np.argmax(self.model.predict(state))
        elif choice == "random":
            action_result = random.randint(0, 2)
        else:
            action_result = 0
            print("enter wrong value")
        return action_result

    def train(self, episodes: int, max_iterations: int) -> list[float]:
        """training function.
        Train for ’episodes’ iterations, where at most ’max_iterations‘ iterations
        should be run for each episode. Returns a list of returns.
        """
        reward_list = []
        for i in range(episodes):
            state = self.env.reset()
            reward_sum = 0
            for j in range(max_iterations):
                action = self.get_action(state)
                (state_new, reward, done) = self.env.step(action)
                reward_sum += reward
                target = reward + self.gamma * np.max(self.model.predict(state_new))
                self.model.update(state, action, target)
                state = state_new
                if done:
                    break
            reward_list.append(reward_sum)
        return reward_list


if __name__ == '__main__':
    # run parser and get parameters values
    args = parser.parse_args()
    environment = MountainCar(mode=args.mode)
    agent = QLearningAgent(environment, mode=args.mode, gamma=args.gamma, epsilon=args.epsilon, lr=args.learning_rate)
    returns = agent.train(args.episodes, args.max_iterations)
    weights = agent.model.weights.reshape(1, -1).tolist()[0]
    weights.insert(0, agent.model.bias)
    write_file(returns, args.returns_out, 'w')
    write_file(weights, args.weight_out, 'w')
