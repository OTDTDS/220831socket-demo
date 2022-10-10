# 简单传输
# 主服务器，调用不同的py
# 先收recvfrom再发sendto
# 1创建服务器套接字（ss = socket()）
# 2绑定服务器套接字（ss.bind()）
# 3服务器无限循环（inf_loop:）
# 4对话（接收与发送）（cs = ss.recvfrom()/ss.sendto()）
# 5关闭服务器套接字（ss.close()）（可选）
# 和c++ cllient UDP连接通信
import os
import numpy as np
import tensorflow as tf
import our_a3c as a3c
import socket
from time import ctime
import struct
import logging

MIN_BIT_RATE = 321
MAX_BIT_RATE = 4280
FEEDBACK_DURATION = 1000
MILLISECONDS_IN_SECOND = 1000
S_INFO = 6
S_LEN = 8
A_DIM = 6
TRAIN_SEQ_LEN = 100
BIT_RATE_INTERVAL = 50
M_IN_K = 1000
REBUF_PENALTY = 20  # 4.0 / (6 / 30)
SMOOTH_PENALTY = 1
DELAY_PENALTY = 10  # //4.0 / 0.4
MODIFY_BIT_RATE = [-1, -450, 0, 50, 100, 200]

instanceId_list = []

MODEL_SAVE_INTERVAL = 5
NN_MODEL = 'E:/bitrate_adaptor_online_learning_20210621/sim_test/echo/python/Pretrained_Model/nn_model_ep_214000.ckpt'
SUMMARY_DIR = 'E:/bitrate_adaptor_online_learning_20210621/sim_test/echo/python/Training_Models/'

RAND_RANGE = 1000
DEFAULT_ACTION = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# add
total_batch_len = 0.0
total_reward = 0.0
total_td_loss = 0.0
# total_entropy = 0.0
# total_agents = 0.0
# min_reward = 0.0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # -1是CPU

tf.Graph().as_default()
global sess
sess = tf.Session(config=config)

actor = a3c.ActorNetwork(sess,
                         state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                         learning_rate=ACTOR_LR_RATE)

critic = a3c.CriticNetwork(sess,
                           state_dim=[S_INFO, S_LEN],
                           learning_rate=CRITIC_LR_RATE)

# add
summary_ops, summary_vars = a3c.build_summaries()
writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

# save neural net parameters
saver = tf.train.Saver()
saver.restore(sess, NN_MODEL)
print("Model restored. %s " % NN_MODEL)
sess.run(tf.global_variables_initializer())  ##


# 更新state, s_batch
def state_reward_batch_update(state, s_batch, rb_batch, r_batch, iter_count, sending_bit_rate,
                              buffer_size, received_bit_rate, delay, packet_loss_rate, nack_sent_count):
    for i in range(1, S_LEN):
        for j in range(0, S_INFO):
            state[j, i - 1] = state[j, i]
    # s_batch = np.roll(s_batch, -1, axis=1)
    # delay = delay / MILLISECONDS_IN_SECOND
    rebuf = max(0, 1 - buffer_size)
    if (delay < 1e-4):
        delay = 1e-4
    if (packet_loss_rate < 1e-4):
        packet_loss_rate = 1e-4
    # 更新状态值
    state[0, S_LEN - 1] = min(1, 2 * (sending_bit_rate - (MIN_BIT_RATE + MAX_BIT_RATE) / 2) / float(
        MAX_BIT_RATE - MIN_BIT_RATE))
    state[1, S_LEN - 1] = 2 * (rebuf / (FEEDBACK_DURATION / MILLISECONDS_IN_SECOND) - 0.5)
    state[2, S_LEN - 1] = 2 * (received_bit_rate - (MAX_BIT_RATE + MIN_BIT_RATE) / 2) / float(
        MAX_BIT_RATE - MIN_BIT_RATE)
    state[3, S_LEN - 1] = 2 * (np.log10(delay) / 4 + 0.5)
    state[4, S_LEN - 1] = 2 * (np.log10(packet_loss_rate) / 4 + 0.5)
    state[5, S_LEN - 1] = np.log10(float(nack_sent_count) + 1) - 1
    # print(state[0, S_LEN - 1], state[1, S_LEN - 1], state[2, S_LEN - 1], state[3, S_LEN - 1], state[4, S_LEN - 1],
    #       state[5, S_LEN - 1])
    # 加入到s_batch中
    for i in range(0, S_INFO):
        for j in range(0, S_LEN):
            s_batch[iter_count % TRAIN_SEQ_LEN][i][j] = state[i][j]

    # 加入到rb_batch中
    rb_batch[iter_count % TRAIN_SEQ_LEN] = received_bit_rate
    # 计算reward，并加入r_batch中
    bitrate = received_bit_rate / M_IN_K
    if iter_count == 0:
        last_bitrate = 0
    elif iter_count % TRAIN_SEQ_LEN == 0:
        last_bitrate = rb_batch[TRAIN_SEQ_LEN - 1] / M_IN_K  # 和c++不一样
    else:
        last_bitrate = rb_batch[iter_count % TRAIN_SEQ_LEN - 1] / M_IN_K

    reward = bitrate - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * abs(bitrate - last_bitrate) - DELAY_PENALTY * delay
    r_batch[iter_count % TRAIN_SEQ_LEN] = reward
    pylogfile.write("reward = %s,iter_count = %s\n" % (reward, iter_count))

    return state, s_batch, rb_batch, r_batch


def action_set(state, a_batch, iter_count):  # 码率切换
    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)  # 累加
    action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    # pylogfile.write(str(state) + '\n')
    if (np.isnan(action_prob[0][0]) or np.isinf(action_prob[0][0])):
        print('gradient burst,rollback...')
        pylogfile.write('gradient burst,rollback...\n')
        # saver.restore(sess, NN_MODEL)
        return_action = DEFAULT_ACTION
        # initial_model_flag = True
    else:
        return_action = action

    action_cb = MODIFY_BIT_RATE[return_action]
    pylogfile.write("action_cb=%s\n" % action_cb)
    action_vec = np.zeros(A_DIM)
    action_vec[return_action] = 1
    # 加入到a_batch
    for i in range(0, A_DIM):
        a_batch[iter_count % TRAIN_SEQ_LEN][i] = action_vec[i]
    return action_cb, a_batch


def bitrate_judge(action, bitrate, packet_loss_rate):
    if (action == -1):
        if (packet_loss_rate <= 1e-4):
            bitrate = int(np.floor(
                float((1 - packet_loss_rate + 1e-4) * bitrate) / float(BIT_RATE_INTERVAL)) * BIT_RATE_INTERVAL)
        else:
            bitrate = int(np.floor(
                float((1 - packet_loss_rate - 0.1) * bitrate) / float(BIT_RATE_INTERVAL)) * BIT_RATE_INTERVAL)
    else:
        bitrate = action + bitrate

    if (bitrate < MIN_BIT_RATE):
        bitrate = MIN_BIT_RATE
    if (bitrate > MAX_BIT_RATE):
        bitrate = MAX_BIT_RATE

    return bitrate


def AbsDocPath(docname):
    base_dir = SUMMARY_DIR
    if (not os.path.exists(base_dir)):
        os.makedirs(base_dir)
    return base_dir + docname


def online_learning(a_batch, r_batch, s_batch, epoch):
    pylogfile = open('online_learning.log', 'a+')
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

    # 数据
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
        pylogfile.write('New Model Saved online_nn_model_ep_%s.ckpt \n' % epoch)
        # Check(fp)
    pylogfile.close()  # 关闭


def close():
    sess.close()
    # fp.close()
    print("close session")


def main():

    # 初始化
    state = np.zeros([S_INFO, S_LEN], dtype=float)
    s_batch = np.zeros([TRAIN_SEQ_LEN, S_INFO, S_LEN], dtype=float)  # s_batch,state
    a_batch = np.zeros([TRAIN_SEQ_LEN, S_INFO], dtype=int)  # ac·tion
    r_batch = np.full((TRAIN_SEQ_LEN,), 0, dtype=float)
    rb_batch = np.zeros(TRAIN_SEQ_LEN, dtype=float)  # received_bit_rate

    HOST = ''
    PORT = 9971  # 修改
    BUFSIZ = 1024
    client_addr = (HOST, PORT)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP连接
    server_sock.bind(client_addr)  # 绑定服务器地址

    while True:  # 服务器无线循环
        # payload_fmt = struct.Struct('QiIfffI')  # 数据包
        print('等待连接...')
        recv_str, client_addr = server_sock.recvfrom(BUFSIZ)  # 接受客户的连接
        # server_sock.sendto(bytes('[%s] %s' % (ctime(), recv_str), encoding='UTF-8'), client_addr)  # 发送UDP 数据

        # a,b,c,d,e,f= struct.unpack('!QiIfffI', recv_str)
        iter_count, sending_bit_rate, received_bit_rate, buffer_size, \
        delay, packet_loss_rate, nack_sent_count = struct.unpack('QiIfffI', recv_str)

        # data = payload_fmt.unpack(recv_str)
        # iter_count = int(data[0])
        # int = int(data[1])
        # received_bit_rate = int(data[2])
        # buffer_size = float(data[3])
        # delay = float(data[4])
        # packet_loss_rate = float(data[5])
        # nack_sent_count = int(data[6])
        print('iter_count:' + str(iter_count) + '\n'
              + 'sending_bit_rate:' + str(sending_bit_rate) + '\n'
              + 'received_bit_rate:' + str(received_bit_rate) + '\n'
              + 'buffer_size:' + str(buffer_size) + '\n'
              + 'delay:' + str(delay) + ' ms\n'
              + 'packet_loss_rate:' + str(packet_loss_rate) + '\n'
              + 'nack_sent_count:' + str(nack_sent_count) + '\n')

        pylogfile.write("bitrate = %s Kb/s\n" % sending_bit_rate)

        # 1.从c++获得信息！！6个状态+iter_count
        state, s_batch, rb_batch, r_batch = state_reward_batch_update(state, s_batch, rb_batch, r_batch, iter_count,
                                                                      sending_bit_rate, buffer_size, received_bit_rate,
                                                                      delay, packet_loss_rate, nack_sent_count)
        action, a_batch = action_set(state, a_batch, iter_count)
        pylogfile.write("action = %s Kb/s\n" % action)

        bitrate_cb = bitrate_judge(action, sending_bit_rate, packet_loss_rate)
        if iter_count % TRAIN_SEQ_LEN == 0:
            epoch = iter_count / TRAIN_SEQ_LEN
            pylogfile.write("epoch = %s\n" % epoch)
            online_learning(a_batch, r_batch, s_batch, epoch)

        pylogfile.write("bitrate_cb = %s Kb/s\n" % bitrate_cb)

        # 7 返回客户端我的码率,RL反馈包
        # SD = struct.Struct('II')
        # senddata = SD.pack(instanceId, bitrate_cb)
        senddata = struct.pack('i', bitrate_cb)
        # 8 向客户端传输RL反馈包，把码率反馈给他们
        # print('senddata:' + str(senddata))
        server_sock.sendto(senddata, client_addr)  # 发送UDP 数据
        print('bitrate_cb:' + str(bitrate_cb))
        print('连接地址:', client_addr)
        # print('内容是：', recv_str.decode("GBK"))

    # 9 退出套接字,关闭服务器连接
    connection.close()
    # server_sock.close()  # 关闭服务器连接
    pylogfile.close()


# python带入数据测试（只有python测试，测试已经成功）
def test():
    pylogfile = open('bitrate_adaptor.log', 'w+')
    # 测试输入数据bitrate
    bitrate = 1983  # kb/s
    pylogfile.write("bitrate = %s Kb/s\n" % bitrate)

    state = np.zeros([S_INFO, S_LEN], dtype=float)
    s_batch = np.zeros([TRAIN_SEQ_LEN, S_INFO, S_LEN], dtype=float)  # s_batch,state
    a_batch = np.zeros([TRAIN_SEQ_LEN, S_INFO], dtype=int)  # ac·tion
    r_batch = np.full((TRAIN_SEQ_LEN,), 0, dtype=float)
    rb_batch = np.zeros(TRAIN_SEQ_LEN, dtype=float)  # received_bit_rate
    for i in range(600):
        # 测试输入数据
        iter_count = i + 1
        sending_bit_rate = 1983  # kb/s
        buffer_size = 0.900000  # fp
        received_bit_rate = 4665  # kb/s
        delay = 0.010000  # ms
        packet_loss_rate = 0.00000  # loss
        nack_sent_count = 0  # N
        # 1.从c++获得信息！！6个状态+iter_count
        state, s_batch, rb_batch, r_batch = state_reward_batch_update(state, s_batch, rb_batch, r_batch, iter_count,
                                                                      sending_bit_rate, buffer_size, received_bit_rate,
                                                                      delay, packet_loss_rate, nack_sent_count)
        action, a_batch = action_set(state, a_batch, iter_count)
        pylogfile.write("action = %s Kb/s\n" % action)

        bitrate_cb = bitrate_judge(action, bitrate, packet_loss_rate)
        if iter_count % TRAIN_SEQ_LEN == 0:
            epoch = iter_count / TRAIN_SEQ_LEN
            pylogfile.write("epoch = %s\n" % epoch)  # 重复了！！
            online_learning(a_batch, r_batch, s_batch, epoch)

        pylogfile.write("bitrate_cb = %s Kb/s\n" % bitrate_cb)
        # 2.把bitrate传给c++
    pylogfile.close()


if __name__ == '__main__':
    pylogfile = open('bitrate_adaptor.log', 'w+')
    main()
