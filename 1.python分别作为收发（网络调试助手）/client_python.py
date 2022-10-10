#UDP客户端client,和网络调试助手连接

import socket
def main():
    # 创建一个套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        # 从键盘获取数据
        send_data = input("请输入要发送的数据：")
        # 退出函数
        if send_data == "exit":
            break
        # 可以使用套接字收发数据,此时未绑定发送的端口号，系统每次会随机分配一个
        # udp_socket.sendto("hahaha",对方的IP和port)
        udp_socket.sendto(send_data.encode("gbk"),("192.168.31.57",54679))#相互交互
        # udp_socket.sendto(send_data.encode("gbk"), ("192.168.31.57", 9991))
        # #9991为网络调试助手本地端口
        #由于接收是在Windows上，而Windows中默认编码为gbk
        #8080 本地主机port

    # 关闭套接字
    udp_socket.close()
if __name__ == '__main__':
    main()
