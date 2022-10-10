# 先收recvfrom再发sendto
# 1创建服务器套接字（ss = socket()）
# 2绑定服务器套接字（ss.bind()）
# 3服务器无限循环（inf_loop:）
# 4对话（接收与发送）（cs = ss.recvfrom()/ss.sendto()）
# 5关闭服务器套接字（ss.close()）（可选）

#和python cllient UDP连接通信
import socket
from time import ctime

HOST = ''
PORT = 8077
BUFSIZ = 1024
ADDR = (HOST, PORT)

udpServerSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP连接
udpServerSocket.bind(ADDR)  # 绑定服务器地址

while True:  # 服务器无线循环
    print('等待连接...')
    data, addr = udpServerSocket.recvfrom(BUFSIZ)  # 接受客户的连接
    # udpServerSocket.sendto(bytes('[%s] %s' % (ctime(), data), encoding='utf-8'), addr)  # 发送UDP 数据
    udpServerSocket.sendto(data, addr)  # 发送UDP 数据
    print('连接client地址:', addr)
    print('server接收内容是：',data.decode("UTF-8"))

udpServerSocket.close()  # 关闭服务器连接