import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import scipy
from scipy.stats import logistic


PATH = '../input/btcxbt/bitstamp_15-min_wo_nan.csv'

df = pd.read_csv(PATH)
data = df['Close'].tolist()
btc_vol = df['Volume_(BTC)'].tolist()
btc_ini_balance = btc_vol[0]

class DQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=300)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 1e-3
        self.learning_rate = 0.003
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.start_reduce_epsilon = 200
        

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        
        #model.add(LSTM(64,
                       #input_shape=(1,4),
                       #return_sequences=False,
        #               stateful=False
        #               ))
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)

    def update_epsilon(self, total_step):
        if self.epsilon > self.epsilon_min and total_step > self.start_reduce_epsilon:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def formatPrice(n):
    return("-Rs." if n<0 else "Rs.")+"{0:.2f}".format(abs(n))

window_size = 20
episode_count = 5

agent = DQNAgent(window_size)

l = len(data) - 1
batch_size = 64

# inventory=[]

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(logistic.cdf(block[i + 1] - block[i]))
    return np.array([res])



for e in range(episode_count + 1):
    
    print("Episode " + str(e) + "/" + str(episode_count))
    
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    
    for t in range(l):
        action = agent.act(state)
        # hold
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        
        if action == 1: # buy
            agent.inventory.append(data[t])
#             data[t]+=btc_ini_balance
            print("Buy: " + formatPrice(data[t]))
        
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
#             data[t]-=btc_ini_balance
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        
        done = True if t == l - 1 else False
        
        #print((state, action, reward, next_state, done))
        
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        
        if done:
            print("Total Profit: " + formatPrice(total_profit))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if e % 10 == 0:
        agent.save(str(e))
        
        