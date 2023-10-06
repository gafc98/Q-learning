import gym
import numpy as np
import network

STATE_SPACE_GRID_DIV = [20, 20]
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPISODES = 4000
RENDER_MODE = "human"

SHOW_EVERY = 500

env = gym.make("MountainCar-v0", render_mode = "human")
env.reset()

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)

print(env.observation_space.high)
print(env.observation_space.low)
print(type(env.observation_space.low))

def discretize_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return discrete_state


q_table = np.random.uniform(low=-2, high=0, size=(STATE_SPACE_GRID_DIV + [env.action_space.n]))




nt = network.Network([2, 10, 10, 3])


print(q_table[19, 19])

for episode in range(1, EPISODES+1):
    render = False
    if episode % SHOW_EVERY == 0:
        render_mode = RENDER_MODE
    else:
        render_mode = None

    env = gym.make("MountainCar-v0", render_mode = render_mode)

    state, info = env.reset()
    state = discretize_state(state)
    done = False
    max_iter = 20000
    it = 0
    score = 0
    while not done and it < max_iter:
        curr_q = nt.Feedforward(state)
        action = np.argmax(curr_q)
        new_state, reward, done, _, info = env.step(action)
        new_state = discretize_state(new_state)

        #added reward for moving closer
        reward += 0.1 * new_state[0]/STATE_SPACE_GRID_DIV[0]

        table_entry = state + (action, )
        if not done:
            #q_table[table_entry] = (1 - LEARNING_RATE)* q_table[table_entry] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[new_state]))
            q_target = reward + DISCOUNT_FACTOR * np.max(curr_q)
            curr_q[action] = q_target
            #print('lmao')
            #print([list(curr_q)])
            nt.Train(epoch=1, input_data=[list(state)], target_data=[list(curr_q)], batch_size=1, learning_rate=LEARNING_RATE, show_results=False, autosave=False, print_progress=False)
        elif new_state[0] > env.goal_position:
            pass#q_table[table_entry] = 0

        state = new_state

        it += 1
        score += reward
    
    print(f"Episode: {episode}\tScore: {score}")
    env.close()
env.close()