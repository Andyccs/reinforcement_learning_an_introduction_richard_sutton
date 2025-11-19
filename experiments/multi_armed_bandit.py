import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandits:
    def __init__(self, num_arms, mu=None, sigma=None):
        self.num_arms = num_arms
        if mu is None:
            self.mu = np.random.normal(0, 1.0, num_arms)
        else:
            self.mu = np.array(mu)

        if sigma is None:
            self.sigma = np.ones(num_arms)
        else:
            self.sigma = np.array(sigma)

    def pull_arm(self, arm_index):
        if not (0 <= arm_index < self.num_arms):
            raise ValueError("Arm index out of bounds.")

        reward = np.random.normal(self.mu[arm_index], self.sigma[arm_index])
        return reward

    def reset(self):
        # In this simple bandit, reset doesn't change the underlying reward distributions
        # but in more complex environments, it might reset state or other parameters.
        pass

    def plot_arm_distributions(self):
        x = np.arange(self.num_arms)
        plt.errorbar(x, self.mu, yerr=self.sigma, fmt="o", capsize=5)
        plt.xlabel("Arm Index")
        plt.ylabel("Mean Reward (with Std Dev)")
        plt.title("Multi-Armed Bandit Arm Distributions")
        plt.xticks(x)
        plt.grid(True)
        plt.show()


def simulate_env(
    *, env, initial_q_value=0, episodes=1000, epsilon=0.1, alpha=None, ucb_c=None
):
    q_values = np.ones(env.num_arms) * initial_q_value
    action_counts = np.zeros(env.num_arms)
    rewards = np.zeros(episodes)
    is_optimal = np.zeros(episodes)
    for i in range(episodes):
        if ucb_c:
            ucb_values = q_values + ucb_c * np.sqrt(
                np.log(i + 1) / (action_counts + 1e-5)
            )
            action = np.argmax(ucb_values)
        else:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.num_arms)
            else:
                action = np.argmax(q_values)

        reward = env.pull_arm(action)
        action_counts[action] += 1
        increment_q = reward - q_values[action]
        if alpha is None:
            increment_q = increment_q / action_counts[action]
        else:
            increment_q = increment_q * alpha
        q_values[action] += increment_q
        rewards[i] = reward
        is_optimal[i] = int(action == np.argmax(env.mu))
    return q_values, rewards, is_optimal


def simulates(*, n=2000, epsilon=0.1, initial_q_value=0, alpha=None, ucb_c=None):
    rewards = np.zeros((n, 1000))
    optimals = np.zeros((n, 1000))
    for i in range(n):
        env = MultiArmedBandits(10)
        q, r, o = simulate_env(
            env=env,
            initial_q_value=initial_q_value,
            epsilon=epsilon,
            alpha=alpha,
            ucb_c=ucb_c,
        )
        rewards[i] = r
        optimals[i] = o
    avg_rewards = np.mean(rewards, axis=0)
    avg_optimals = np.sum(optimals, axis=0) / n
    return avg_rewards, avg_optimals


def plot_avg_rewards(list_of_avg_rewards, labels):
    for avg_rewards, label in zip(list_of_avg_rewards, labels):
        plt.plot(avg_rewards, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_avg_optimals(list_of_avg_optimals, labels):
    for avg_optimals, label in zip(list_of_avg_optimals, labels):
        plt.plot(avg_optimals, label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("% Optimal Action over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()


def gradient_bandit_simulate_env(*, env, episodes=1000, alpha=0.1, use_baseline=True):
    def softmax(x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    h = np.zeros(env.num_arms)
    rewards = np.zeros(episodes)
    mean_reward = 0
    is_optimal = np.zeros(episodes)
    for i in range(episodes):
        pi = softmax(h)
        action = np.random.choice(env.num_arms, p=pi)
        reward = env.pull_arm(action)

        baseline = mean_reward if use_baseline else 0

        h = h + alpha * (reward - baseline) * ((np.arange(env.num_arms) == action) - pi)
        mean_reward += (reward - mean_reward) / (i + 1)

        rewards[i] = reward
        is_optimal[i] = int(action == np.argmax(env.mu))
    return h, rewards, is_optimal


def gradient_bandit_simulates(*, n=2000, alpha=0.1, use_baseline=True):
    rewards = np.zeros((n, 1000))
    optimals = np.zeros((n, 1000))
    for i in range(n):
        env = MultiArmedBandits(10, mu=np.random.normal(3.5, 4.5, 10))
        h, r, o = gradient_bandit_simulate_env(
            env=env,
            alpha=alpha,
            use_baseline=use_baseline,
        )
        rewards[i] = r
        optimals[i] = o
    avg_rewards = np.mean(rewards, axis=0)
    avg_optimals = np.sum(optimals, axis=0) / n
    return avg_rewards, avg_optimals
