import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization
import xlsxwriter
import pandas as pd
import openpyxl


#####################  hyper parameters  ####################
MAX_EPISODES = 1000
# MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.999  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
OUTPUT_GRAPH = False


class TD3(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            # First critic network
            q1 = self._build_c(self.S, self.a, scope='eval1', trainable=True)
            q1_ = self._build_c(self.S_, a_, scope='target1', trainable=False)

            # Second critic network
            q2 = self._build_c(self.S, self.a, scope='eval2', trainable=True)
            q2_ = self._build_c(self.S_, a_, scope='target2', trainable=False)

        # Target net replacements
        self.soft_replace_actor = [tf.assign(t, (1 - TAU) * t + TAU * e)
                                   for t, e in zip(tf.trainable_variables('Actor/target'),
                                                   tf.trainable_variables('Actor/eval'))]

        self.soft_replace_critic1 = [tf.assign(t, (1 - TAU) * t + TAU * e)
                                     for t, e in zip(tf.trainable_variables('Critic/target1'),
                                                     tf.trainable_variables('Critic/eval1'))]

        self.soft_replace_critic2 = [tf.assign(t, (1 - TAU) * t + TAU * e)
                                     for t, e in zip(tf.trainable_variables('Critic/target2'),
                                                     tf.trainable_variables('Critic/eval2'))]

        # Target Q value for critic updates
        min_q_target = tf.minimum(q1_, q2_)
        q_target = self.R + GAMMA * min_q_target

        # Critic losses
        q1_loss = tf.losses.mean_squared_error(labels=q_target, predictions=q1)
        q2_loss = tf.losses.mean_squared_error(labels=q_target, predictions=q2)
        self.critic_loss = q1_loss + q2_loss

        # Critic optimization
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.critic_loss,
                                                            var_list=tf.trainable_variables('Critic/eval1') +
                                                            tf.trainable_variables('Critic/eval2'))
        POLICY_UPDATE_DELAY = 2
        # Delayed policy updates
        if self.pointer % POLICY_UPDATE_DELAY == 0:
            a_loss = -tf.reduce_mean(q1)
            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(
                a_loss, var_list=tf.trainable_variables('Actor/eval'))

        # Initialize session
        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    # ... (other methods)
    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        self.sess.run(self.soft_replace_actor)
        self.sess.run(self.soft_replace_critic1)
        self.sess.run(self.soft_replace_critic2)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})

        # Update critic networks
        self.sess.run(self.ctrain, {self.S: bs,
                                    self.a: ba, self.R: br, self.S_: bs_})

    # Build actor
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(
                s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(
                net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(
                net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(
                net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    # Build Critic
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(
                net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(
                net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


np.random.seed(1)
tf.set_random_seed(1)

env = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound  # [-1,1]

td3 = TD3(a_dim, s_dim, a_bound)

# var = 1  # control exploration
var = 0.01  # control exploration
t1 = time.time()
ep_reward_list = []
ep_delay_list = []

s_normal = StateNormalization()
delay_list = ['delays']
episode_list = ['episodes']

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    ep_delay = 0
    j = 0
    while j < MAX_EP_STEPS:
        # Add exploration noise
        a = td3.choose_action(s_normal.state_normal(s))
        # 高斯噪声add randomness to action selection for exploration
        a = np.clip(np.random.normal(a, var), *a_bound)
        s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist, delay = env.step(
            a)
        if step_redo:
            continue
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1
        td3.store_transition(s_normal.state_normal(
            s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍

        if td3.pointer > MEMORY_CAPACITY:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            td3.learn()
        s = s_
        ep_reward += r
        ep_delay += delay
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' %
                  ep_reward, 'Explore: %.3f' % var, 'Delay: %7.2f' %
                  ep_delay)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            ep_delay_list = np.append(ep_delay_list, ep_delay)
            
            delay_list.append(ep_delay)
            episode_list.append(i)
            workbook = xlsxwriter.Workbook("delay_episodes_TD3.xlsx")
            worksheet = workbook.add_worksheet()
            for row_num, item in enumerate(episode_list):
                worksheet.write(row_num, 0, item)
            for row_num, item in enumerate(delay_list):
                worksheet.write(row_num, 1, item)
            workbook.close()
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            file_name = 'output.txt'
            with open(file_name, 'a') as file_obj:
                # 本episode结束
                file_obj.write("\n======== This episode is done ========")
            break
        j = j + 1

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
plt.plot(ep_delay_list)
plt.xlabel("Episode")
plt.ylabel("Delay")
plt.show()



























