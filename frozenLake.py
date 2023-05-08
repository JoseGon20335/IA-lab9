import numpy as np
import gymnasium as gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time


def main():

    epsilon = 1.0
    trainCicle = 10000
    testCicle = 300

    frozenLakeGame = gymnasium.make("FrozenLake-v1", render_mode="human",
                                    desc=generate_random_map(size=4), is_slippery=True)
    frozenLakeGame.metadata["render_fps"] = 30

    q_table = np.zeros((frozenLakeGame.observation_space.n,
                        frozenLakeGame.action_space.n))

    for i in range(trainCicle):
        state = frozenLakeGame.reset()[0]
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = frozenLakeGame.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, _, _ = frozenLakeGame.step(action)

            if done and reward == 0:
                modified_reward = -0.5
            else:
                modified_reward = reward - 0.01

            q_table[state, action] += 0.1 * (
                modified_reward
                + 0.99 * np.max(q_table[next_state, :])
                - q_table[state, action]
            )
            state = next_state

        epsilon = max(0.01, epsilon * 1.0)

        print("Cicle: ", i + 1)

    wins = 0
    iterationInfo = []
    cantIterations = 0

    for i in range(testCicle):
        print(f"Iteration no. {i + 1}")
        cantIterations += 1

        state = frozenLakeGame.reset()[0]
        frozenLakeGame.render()
        done = False

        while not done:
            action = np.argmax(q_table[state, :])
            next_state, reward, done, _, _ = frozenLakeGame.step(action)
            state = next_state
            frozenLakeGame.render()
            time.sleep(0.5)

            if done:
                if reward == 1:
                    print("Agente logro el objetivo + 1")
                    wins += 1
                    iterationInfo.append(cantIterations)
                    cantIterations = 0
                else:
                    print("Agente no logro el objetivo")

    print(f"RESULTADOS: ", wins)
    for x in range(len(iterationInfo)):
        print(f"{x + 1}: {iterationInfo[x]}")

    frozenLakeGame.close()


main()
