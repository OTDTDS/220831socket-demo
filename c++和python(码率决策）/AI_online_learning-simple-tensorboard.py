"""hg-tensorboard"""
import os
import numpy as np
import tensorflow as tf
import our_a3c as a3c
import time
import datatime
import socket
import struct
import logging

S_INFO = 6
S_LEN = 8
A_DIM = 6

MODEL_SAVE_INTERVAL = 5


NN_MODEL = 'E:/bitrate_adaptor_online_learning_20210621/sim_test/echo/python/Pretrained_Model/nn_model_ep_214000.ckpt'
SUMMARY_DIR = 'E:/bitrate_adaptor_online_learning_20210621/sim_test/echo/python/Training_Models/'

RAND_RANGE = 1000
DEFAULT_ACTION = 2

# add
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

#add
total_batch_len = 0.0
total_reward = 0.0
total_td_loss = 0.0
# total_entropy = 0.0
# total_agents = 0.0
# min_reward = 0.0

#add

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = "0"#-1是CPU

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)#仅在需要时申请显存空间

tf.Graph().as_default()
global sess
sess = tf.Session(config=config)

actor = a3c.ActorNetwork(sess,
                         state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                         learning_rate=ACTOR_LR_RATE)

critic = a3c.CriticNetwork(sess,
                           state_dim=[S_INFO, S_LEN],
                           learning_rate=CRITIC_LR_RATE)

#add
summary_ops, summary_vars = a3c.build_summaries()
writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

# save neural net parameters
saver = tf.train.Saver()
saver.restore(sess, NN_MODEL)
print("Model restored. %s " % NN_MODEL)
sess.run(tf.global_variables_initializer())##

'''
input:state[6,8]
return:action,1-6之间
和c++有个交互
'''
def bitrate_changed(state):  # 码率切换
    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)  # 累加
    action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    # pylogfile.write(str(state) + '\n')
    if (np.isnan(action_prob[0][0]) or np.isinf(action_prob[0][0])):
        print('gradient burst,rollback...')
        pylogfile.write('gradient burst,rollback...\n')
        #saver.restore(sess, NN_MODEL)
        return_action = DEFAULT_ACTION
        # initial_model_flag = True
    else:
        return_action = action
    pylogfile.close()
    return return_action


def AbsDocPath(docname):
    base_dir = SUMMARY_DIR
    if (not os.path.exists(base_dir)):
        os.makedirs(base_dir)
    return base_dir + docname


def online_learning(a_batch, r_batch, s_batch, epoch):
    pylogfile = open('learning.log', 'r+')
    pylogfile.write("epoch = %s \n" % epoch)

    # add
    total_batch_len = 0.0
    total_reward = 0.0
    total_td_loss = 0.0
    # total_entropy = 0.0
    # total_agents = 0.0

    pylogfile.write('Train Process Start...\n')
    print('Train Process Start...')  #
    actor_gradient, critic_gradient, td_batch = \
        a3c.compute_gradients(np.stack(s_batch, axis=0),
                              np.vstack(a_batch),
                              np.vstack(r_batch),
                              True,
                              actor,
                              critic)
    actor.apply_gradients(actor_gradient)
    critic.apply_gradients(critic_gradient)

    #数据
    # if (np.mean(r_batch) < min_reward):
    #     min_reward = np.mean(r_batch)
    total_reward += np.mean(r_batch)
    total_td_loss += np.mean(td_batch)
    total_batch_len += len(r_batch)

    avg_reward = total_reward
    avg_td_loss = total_td_loss / total_batch_len

    summary_str = sess.run(summary_ops, feed_dict={
        summary_vars[0]: avg_td_loss,
        summary_vars[1]: avg_reward,
    })
    writer.add_summary(summary_str, epoch)
    writer.flush()
     ###
    print('Train Process End...')
    pylogfile.write('Train Process End...\n')

    if epoch % MODEL_SAVE_INTERVAL == 0:
        saver.save(sess, AbsDocPath("online_nn_model_ep_" +
                                    str(epoch) + ".ckpt"))
        print('New Model Saved online_nn_model_ep_%s.ckpt ' % epoch)
        pylogfile.write('New Model Saved online_nn_model_ep_%s.ckpt \n ' % epoch)
        # Check(fp)
    pylogfile.close()  # 关闭


def close():
    sess.close()
    # fp.close()
    print("close session")

def main():
    HOST = ''
    PORT = 9992
    BUFSIZ = 1024
    client_addr = (HOST, PORT)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP连接
    server_sock.bind(client_addr)  # 绑定服务器地址

    for i in range(5):
        epoch = i + 1
        # online_learning(a_batch, r_batch, s_batch, epoch)
            online_learning(server_sock, epoch)
    
# 调试
if __name__ == '__main__':

    s_batch = np.zeros([100, 6, 8])  # s_batch,state
    a_batch = np.zeros([100, 6])  # action
    r_batch = np.full((100,), -6, dtype=float)
    rb_batch = np.zeros(100)  # received_bit_rate

    for i in range(5):
        epoch = i + 1
        online_learning(a_batch, r_batch, s_batch, epoch)
        # time.sleep(10)
    # fp.close()
