#客户端，先发sendto再收recvfrom
# 1创建客户套接字（ss = socket()）
# 2通讯循环（comm_loop:）
# 3对话（接收与发送）（cs.sendto() / cs.recvfrom()）
# 4关闭客户套接字（cs.close()）
#和python server UDP连接通信
import socket

HOST = '192.168.31.57'  # 服务器连接地址
PORT = 8077  # 服务器启用端口
BUFSIZ = 1024  # 缓冲区大小
ADDR = (HOST, PORT)

udpClientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    data = bytes(input('> '), encoding="UTF-8")
    if not data:
        break
    udpClientSocket.sendto(data, ADDR)
    data, ADDR = udpClientSocket.recvfrom(BUFSIZ)
    if not data:
        break
    print('client接收的内容是：',data.decode('UTF-8'))

udpClientSocket.close()