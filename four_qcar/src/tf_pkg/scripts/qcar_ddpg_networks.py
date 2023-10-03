#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.python.keras.engine import training
# import cv2

from gazebo_env_two_qcar_link1 import envmodel1

env = envmodel1()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, max_size, input_x, input_y, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_x, input_y))
        self.new_state_memory = np.zeros((self.mem_size, input_x, input_y))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size,replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=64, fc2_dims=64,
                 name='critic', chkpt_dir='tmp/ddpg', training=True):
        super(CriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_y = 362

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg')

        self.training = training

        f1 = 1. / np.sqrt(self.input_y)
        f2 = 1. / np.sqrt(self.fc1_dims)
        fa = 1. / np.sqrt(1)

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        self.batch1 = layers.BatchNormalization()
        # self.batch2 = layers.BatchNormalization()
        # self.batch3 = layers.BatchNormalization()
        # self.batch4 = layers.BatchNormalization()
        
        self.relu = layers.Activation('relu')

        self.lstm = tf.keras.layers.LSTM(362)

        self.fc1 = Dense(self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                         bias_initializer=tf.random_uniform_initializer(-f1, f1))

        self.action_in = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-fa, fa),
                               bias_initializer=tf.random_uniform_initializer(-fa, fa), activation='relu')

        self.fc2 = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                         bias_initializer=tf.random_uniform_initializer(-f2, f2))

        # self.fc3 = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2),
        #                  bias_initializer=tf.random_uniform_initializer(-f2, f2))

        self.q = Dense(1, activation=None, kernel_initializer=last_init,
                       bias_initializer=last_init, kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, state, action):

        x = self.lstm(state)  # (32,512) + (32,1)
        x = self.batch1(x, training=self.training)
        x = self.fc1(x)
        #x = self.batch2(x, training=self.training)
        x = self.relu(x)
        # x = self.fc2(x)
        x = self.fc2(x)
        x = self.relu(x)
        action_in = self.action_in(action)
        action_value = tf.concat([x, action_in], axis=1)
        #action_value = self.batch3(action_value, training=self.training)
        #action_value = self.relu(action_value)
        #action_value = self.fc3(action_value)
        #action_value = self.batch4(action_value, training=self.training)
        #action_value = self.relu(action_value)

        q = self.q(action_value)

        return q

#fc1= 400 , fc2= 300
class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=64, fc2_dims=64, n_actions=1, name='actor',
                 chkpt_dir='tmp/ddpg', training=True):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg')
        self.training = training
        self.input_y = 362

        f1 = 1. / np.sqrt(self.input_y)
        f2 = 1. / np.sqrt(self.fc1_dims)

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        self.lstm = layers.LSTM(362)
        self.relu = layers.Activation('relu')
        self.sigmoid = layers.Activation('sigmoid')

        self.fc1 = Dense(self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                         bias_initializer=tf.random_uniform_initializer(-f1, f1))
        
        self.batch1 = layers.BatchNormalization()
        self.batch2 = layers.BatchNormalization()
        self.batch3 = layers.BatchNormalization()
        self.batch4 = layers.BatchNormalization()

        self.fc2 = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                         bias_initializer=tf.random_uniform_initializer(-f2, f2))

        self.mu = Dense(self.n_actions,
                        kernel_initializer=last_init, bias_initializer=last_init)

    def call(self, state):

        # print(np.array(state).shape)
        x = self.lstm(state)
        x = self.batch1(x, training=self.training)
        x = self.fc1(x)
        x = self.batch2(x, training=self.training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch3(x, training=self.training)
        x = self.relu(x)
        mu = self.mu(x)
        mu = self.batch4(mu, training=self.training)
        mu = self.sigmoid(mu)

        return mu


class DDPG:
    def __init__(self, input_x=4, input_y=362, alpha=0.0001, beta=0.001,
                 gamma=0.90, n_actions=1, max_size=100000, tau=0.001,
                 batch_size=128, noise=0.2):

        # Define State Space and Action Space
        self.input_x = input_x  # stackFrame
        self.input_y = input_y  # 360 + 2 self state
        self.n_actions = n_actions
        self.max_action = 2.0
        self.min_action = 0.0

        # Define Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.alpha = alpha
        self.beta = beta

        # Define ReplayBuffer Information
        self.memory = ReplayBuffer(max_size, input_x, input_y, n_actions)
        self.batch_size = batch_size

        # Define Training Information
        self.algorithm = 'DDPG'
        self.Number = 'test1'
        self.progress = ''
        self.load_path = '/home/sdcnlab025/ROS_test/three_qcar/src/tf_pkg/scripts/saved_networks/two_qcar_links_2021-06-08_test2/qcar1'
        self.step = 1
        self.score = 0
        self.episode = 0
        self.isTraining = True

        self.ou_noise = OUActionNoise(mean=np.zeros(
            1), std_deviation=float(self.noise) * np.ones(1))

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())
        # Define and Create Saved Location
        self.save_location = 'saved_networks/' + 'two_qcar_links_' + \
            self.date_time + '_' + self.Number + '/qcar1'
        os.makedirs(self.save_location)

        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4
        # input image size
        self.img_size = 80

        # Define Network Information
        self.actor = ActorNetwork(
            n_actions=n_actions, chkpt_dir=self.save_location)
        self.critic = CriticNetwork(chkpt_dir=self.save_location)
        self.target_actor = ActorNetwork(
            n_actions=n_actions, name='target_actor', chkpt_dir=self.save_location)
        self.target_critic = CriticNetwork(
            name='target_critic', chkpt_dir=self.save_location)

        self.actor.compile(optimizer=Adam(lr=alpha))
        self.critic.compile(optimizer=Adam(lr=beta))
        self.target_actor.compile(optimizer=Adam(lr=alpha))
        self.target_critic.compile(optimizer=Adam(lr=beta))

        self.update_network_parameters(tau=1)

        self.init_sess()

        # self.actor._set_inputs([32,4, 362])
        # self.target_actor._set_inputs([32, 4, 362])
        # self.critic._set_inputs([32.4, 362],[32,1])
        # self.target_critic._set_inputs([32, 4, 362], [32,1])

    #@tf.function
    def update_network_parameters(self, tau=None):

        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_load_path = os.path.join(self.load_path, 'actor_ddpg')
        self.target_actor_load_path = os.path.join(
            self.load_path, 'target_actor_ddpg')
        self.critic_load_path = os.path.join(self.load_path, 'critic_ddpg')
        self.target_critic_load_path = os.path.join(
            self.load_path, 'target_critic_ddpg')

        self.actor.load_weights(self.actor_load_path)
        self.target_actor.load_weights(self.target_actor_load_path)
        self.critic.load_weights(self.critic_load_path)
        self.target_critic.load_weights(self.target_critic_load_path)

    def input_initialization(self, env_info):
        state = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.input_y))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)
            # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)

        return observation_stack, observation_set, state_stack, state_set

    # Resize input information
    def resize_input(self, env_info, observation_set, state_set):

        observation = env_info[1]
        observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        del observation_set[0]
        observation_stack = np.uint8(observation_stack)

        state = env_info[0]
        state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.input_y))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        del self.state_set[0]

        return observation_stack, observation_set, state_stack, state_set

    def move(self, cmd=[0.0, 0.0]):
        env.step(cmd)

    def accelerate(self, accel):
        env.accel(accel)

    def update_path(self, path):
        self.path = path
        env.update_path(self.path)

    def init_sess(self):
        # Load the file if the saved file exists
        self.isTraining = True
        check_save = input('Load Model for Link 1? (1=yes/2=no): ')
        if check_save == 1:
            # Restore variables from disk.
            self.load_models()
            print("Link 1 model restored.")

            check_train = input(
                'Inference or Training? (1=Inference / 2=Training): ')
            if check_train == 1:
                self.isTraining = False
                self.Num_start_training = 0
                self.Num_training = 0

    def new_environment(self):

        plt.scatter(self.episode, self.score, c='r')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.xlim(-1, ((self.episode/20 + 1)*20))
        plt.ylim(-12, 3)
        plt.pause(0.01)  # 0.05

        if self.progress != 'Observing' :
            self.reward_list.append(self.score)
            self.reward_array = np.array(self.reward_list)
            # ------------------------------
            np.savetxt(self.save_location + '/qcar1_reward.txt',
                       self.reward_array, delimiter=',')
            self.episode += 1
            
            
        #if self.progress != 'Observing':
            
        if self.progress == 'Testing' and self.episode % 20 == 0:
            avg_score = np.mean(self.reward_list[-50:])
            print('___________Avgerage score is_____________', avg_score)
            self.avg_score.append(avg_score)
            self.avg_array = np.array(self.avg_score)
            np.savetxt(self.save_location + '/qcar1_test_avg_score.txt',
                       self.avg_array, delimiter=',')

            
        self.score = 0
        self.reward = 0
        env_info = env.get_env()
        self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(
            env_info)

    def main_func(self):

        self.reward_list = []
        self.avg_score = []

        np.random.seed(1000)

        tf.random.set_seed(520)

        env_info = env.get_env()
        # env.info为4维，第1维为相机消息，第2维为agent robot的self state，第3维为terminal，第4维为reward
        self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(
            env_info)

        self.step_for_newenv = 0

    def update_parameters(self, Num_start_training, Num_training, Num_test, learning_rate, gamma, MAXEPISODES):

        self.Num_start_training = Num_start_training
        self.Num_training = Num_training
        self.Num_test = Num_test
        self.learning_rate = learning_rate
        self.gamma = gamma

    def get_progress(self, step):

        if step <= self.Num_start_training:
            # Obsersvation
            progress = 'Observing'

        elif step <= self.Num_start_training + self.Num_training and self.isTraining:
            # Training
            progress = 'Training'
            self.actor.training = True
            self.target_actor.training = True
            self.critic.training = True
            self.target_critic.training = True

        elif step < self.Num_start_training + self.Num_training + self.Num_test:
            # Testing
            progress = 'Testing'
            # print('_________________start testing___________________')
            self.actor.training = False
            self.target_actor.training = False
            self.critic.training = False
            self.target_critic.training = False

        else:
            # Finished
            progress = 'Finished'

        self.progress = progress

        return progress

    def save_fig(self):
        plt.savefig(self.save_location + '/qcar1_reward.png')
        plt.show()
# *****************************************Update information*********************************************************************

    def update_information(self):
        # Get information for update
        env_info = env.get_env()

        self.next_observation_stack, self.observation_set, self.next_state_stack, self.state_set = self.resize_input(
            env_info, self.observation_set, self.state_set)  # 调整输入信息
        terminal = env_info[-2]  # 获取terminal
        self.reward = env_info[-1]  # 获取reward

        self.memory.store_transition(
            self.state_stack, self.action, self.reward, self.next_state_stack, terminal)

        if self.progress == 'Training':
            self.learn()

        # Update information
        self.step += 1
        self.score += self.reward
        self.observation_stack = self.next_observation_stack
        self.state_stack = self.next_state_stack
        self.step_for_newenv += 1

        return terminal
# *****************************************Select Action*********************************************************************

    def select_action(self, state_stack, progress, noise_object):

        state = tf.convert_to_tensor([state_stack], dtype=tf.float32)
        # print('state_stack:__________',state_stack)
        actions = self.actor(state)

        noise = noise_object()

        # actions = tf.math.scalar_mul(0.75, actions)
        # actions = tf.add(actions, 0.75)
        actions = tf.math.multiply(actions, self.max_action)

        if progress == 'Training':
            # actions += tf.random.normal(shape=[self.n_actions, 1],
            #                             mean=0.0, stddev=self.noise)
            actions += noise
        elif progress == "Observing":
            actions = tf.random.normal(shape=[self.n_actions, 1],
                                       mean=1.0, stddev=1.0)

        # note that if the environment has an action > 1, we have to multiply by max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        # print('action', actions)
        return actions[0][0].numpy()  # Exact the value

    def return_action(self):
        self.action = self.select_action(
            self.state_stack, self.progress, self.ou_noise)
        return self.action

    #@tf.function
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)

            critic_value_ = tf.squeeze(self.target_critic(
                states_, target_actions), 1)

            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_ * (1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                 self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = - self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def print_information(self):
        print('[Link1-'+self.progress+'] step:'+str(self.step)+'/episode:' +
              str(self.episode)+'/path:'+self.path+'/score:' + str(self.score))


# if __name__ == '__main__':
#     agent = DDPG()
#     agent.main()

