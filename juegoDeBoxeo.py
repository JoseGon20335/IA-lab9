import numpy as np
import gymnasium as gymnasium

# Create the game punch_outironment
punch_out = gymnasium.make('Boxing-v0', render_mode='human')

# Define the Q-table as a dictionary
q_table = {}

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99

# Number of episodes for training
num_episodes = 100

# Number of episodes for testing
test_episodes = 10

# Main loop for episodes
for episode in range(num_episodes):
    
    
    obs = punch_out.reset()

    # not rendering in the train phase
    #punch_out.render()

    # Extract the image from the observation
    img = obs[0]

    # Convert the image to a tuple
    img = tuple(img.flatten())

    # Initialize the total reward for the episode
    total_reward = 0

    # Loop for each step in the episode
    while True:
        # Choose an action based on the epsilon-greedy policy
        if np.random.rand() < epsilon:
            #print("Scorpion")
            action = punch_out.action_space.sample()
        else:
            #print("Subzero")
            # Get the Q-values for the current state
            if img in q_table:
                #print("Liu Kang")
                q_values = q_table[img]
            else:
                # Initialize Q-values for the state if it's the first visit
                q_table[img] = np.zeros(punch_out.action_space.n)
                q_values = q_table[img]

            # Choose the action with the maximum Q-value
            action = np.argmax(q_values)

        # Take the action and get the next observation, reward, termination flag, and info
        next_obs, reward, done, q, w = punch_out.step(action)
        #print("next_obs: ", next_obs)
        #print("reward: ", reward)
        #print("done: ", done)
        #print("q: ", q)

        # Extract the image from the next observation
        next_img = next_obs[0]

        # Convert the image to a tuple
        next_img = tuple(next_img.flatten())

        # Update the Q-table
        if img in q_table:
            q_values = q_table[img]
        else:
            q_table[img] = np.zeros(punch_out.action_space.n)
            q_values = q_table[img]

        if next_img in q_table:
            next_q_values = q_table[next_img]
        else:
            q_table[next_img] = np.zeros(punch_out.action_space.n)
            next_q_values = q_table[next_img]

        # Calculate the new Q-value for the current state and the taken action
        q_values[action] = (1 - alpha) * q_values[action] + alpha * (reward + gamma * np.max(next_q_values))

        # Update the Q-table for the current state
        q_table[img] = q_values

        # Update the current observation
        img = next_img

        # Update the total reward
        total_reward += reward

        # Check if the episode has terminated
        if done:
            break

    # Reduce the exploration rate epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Print the total reward for the episode
    print(f"Episode: {episode + 1}, Reward: {total_reward}")

# Store the Q-table
np.save("q_table.npy", q_table)

wins = 0
lose = 0

# Main loop for test episodes
for episode in range(test_episodes):

    punch_out.reset()

    img = obs[0]
    
    img = tuple(img.flatten())

    total_reward = 0

    while True:
        punch_out.render()

        if img in q_table:
            q_values = q_table[img]
        else:
            action = punch_out.action_space.sample()
            obs, reward, done, q, w, info = punch_out.step(action)
            img = tuple(obs[0].flatten())
            continue

        # Choose the action with the maximum Q-value
        action = np.argmax(q_values)

        # Take the action and get the next observation, reward, termination flag, and info
        next_obs, reward, done, q, w = punch_out.step(action)

        # Extract the image from the next observation
        next_img = next_obs[0]

        # Convert the image to a tuple
        next_img = tuple(next_img.flatten())

        # Update the current observationq
        img = next_img

        # Update the total reward
        total_reward += reward

        # Check if the episode has terminated
        if done:
            break

    # Print the total reward for the episode    
    print(f"Episode: {episode + 1}, Reward: {total_reward}")
    if total_reward > 100: # IE Knockout
        wins += 1
    else:
        lose += 1

# Close the environment 
punch_out.close()

# Final results

print(f"Wins: {wins}, Lose: {lose}")

