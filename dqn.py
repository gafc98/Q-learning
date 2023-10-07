import gym
import numpy as np
import network
import NN


LEARNING_RATE = 0.075
DISCOUNT_FACTOR = 0.95
EPISODES = 4000
RENDER_MODE = "human"

EPOCHS = 5

MINIBATCH = 20

SHOW_EVERY = 20

env = gym.make("MountainCar-v0", render_mode = None)
env.reset()

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)

print(env.observation_space.high)
print(env.observation_space.low)
print(type(env.observation_space.low))

def discretize_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size - 0.5
    return discrete_state



nt = NN.NN_interpolator([100], n_input=2, n_out=3, lr = LEARNING_RATE/EPOCHS) # [30, 25] worked well
nt.set_init_params()


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
    max_iter = 10000
    it = 0
    score = 0

    X = []
    y = []

    while not done and it < max_iter:
        curr_q = nt.eval(state)
        action = np.argmax(curr_q)
        new_state, reward, done, _, info = env.step(action)
        new_state = discretize_state(new_state)


        new_q = nt.eval(new_state)
        if not done or True:
            q_target = reward + DISCOUNT_FACTOR * np.max(new_q)
        else:
            pass
            # q_target = reward
            
        curr_q[action] = q_target

        X.append(state)
        y.append(curr_q)

        if len(X) >= MINIBATCH or done:
            nt.set_train_data(X, y)
            nt.train(n_epoch=MINIBATCH)
            X = []
            y = []
        
        state = new_state

        it += 1
        score += reward
    
    print(f"Episode: {episode}\tScore: {score}")
    env.close()
nt.save_model("params")

env.close()