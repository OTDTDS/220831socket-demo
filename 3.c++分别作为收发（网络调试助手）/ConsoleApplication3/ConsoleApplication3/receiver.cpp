//
// Created by JinxBIGBIG on 2022/8/3.
// server服务器-接收端
//与网络调试助手连接成功，端口为远程主机的端口！！

#include <iostream>
#include <WinSock2.h>
/* inet_ntop() */
#include <WS2tcpip.h>
/*链接静态库*/
#pragma comment (lib,"ws2_32.lib")
using namespace  std;

int main()
{    /*初始化套接字库*/
	WSADATA  wsData;

	int nret = WSAStartup(MAKEWORD(2, 2), &wsData);
	if (nret != 0)
	{
		return nret;
	}

	/*初始化地址结构体 服务器sa, 客户端recSa*/
	sockaddr_in sa, recSa;

	int len = sizeof(sa);

	sa.sin_addr.S_un.S_addr = INADDR_ANY;//自动获取IP地址INADDR_ANY
	//sa.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	/* INADDR_ANY表示不管是哪个网卡接收到数据，只要目的端口是SERV_PORT，就会被该应用程序接收到 */
	sa.sin_family = AF_INET;//使用IPV4地址
	sa.sin_port = htons(62513);//端口

	/*申请监听套接字*/
	SOCKET  sock = socket(AF_INET, SOCK_DGRAM, 0);

	if (sock == INVALID_SOCKET)
	{
		return WSAGetLastError();
	}
	/*绑定套接字*/
	bind(sock, (sockaddr*)&sa, len);

	while (true)
	{
		char buf[1024];

		memset(buf, 0, 1024);
		/*返回接收数据长度nlen,数据buf，客户端地址recSa*/
		int nlen = recvfrom(sock, buf, 1024, 0, (sockaddr*)&recSa, &len);

		if (nlen>0)
		{
			char sIP[20];
			//inet_ntop(AF_INET, &recSa.sin_addr, sIP, 20);
			inet_ntoa(recSa.sin_addr);
			//client_adr = inet_ntoa(recSa.sin_addr);
			printf("Client_address:%s Receive data:%s\n", inet_ntoa(recSa.sin_addr), buf);
			//cout << buf << endl;
			//cout << sIP<<buf << endl;
		}
	}
	/*关闭套接字和套接字库*/
	//closesocket(sock);
	//WSACleanup();
}
