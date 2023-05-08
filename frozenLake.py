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

    tableGame = np.zeros((frozenLakeGame.observation_space.n,
                          frozenLakeGame.action_space.n))

    print("Juego creado")

    for i in range(trainCicle):
        game = frozenLakeGame.reset()[0]
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                jugada = frozenLakeGame.action_space.sample()
            else:
                jugada = np.argmax(tableGame[game, :])

            GameNext, premio, done, _, _ = frozenLakeGame.step(jugada)

            if done and premio == 0:
                tempPremio = -0.5
            else:
                tempPremio = premio - 0.01

            tableGame[game, jugada] += 0.1 * (
                tempPremio
                + 0.99 * np.max(tableGame[GameNext, :])
                - tableGame[game, jugada]
            )
            game = GameNext

        epsilon = max(0.01, epsilon * 1.0)

        print("Ciclo train: ", i + 1)

    wins = 0
    iterationInfo = []
    cantIterations = 0

    for i in range(testCicle):
        print("Ciclo test. ", i + 1)
        cantIterations += 1

        game = frozenLakeGame.reset()[0]
        frozenLakeGame.render()
        done = False

        while not done:
            jugada = np.argmax(tableGame[game, :])
            GameNext, premio, done, _, _ = frozenLakeGame.step(jugada)
            game = GameNext
            frozenLakeGame.render()
            time.sleep(0.5)

            if done:
                if premio == 1:
                    print("Agente logro el objetivo + 1")
                    wins += 1
                    iterationInfo.append(cantIterations)
                    cantIterations = 0
                else:
                    print("Agente no logro el objetivo")

    print(f"RESULTADOS: ", wins)
    for i in range(len(iterationInfo)):
        print("--------------------")
        print("Num interacion:", i)
        print(iterationInfo[i])
        print("--------------------")

    frozenLakeGame.close()


main()
