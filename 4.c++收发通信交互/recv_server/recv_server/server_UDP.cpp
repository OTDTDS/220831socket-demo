//
// Created by JinxBIGBIG on 2022/8/3.
// server服务器

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

	/*初始化地址结构体 服务器addrServ, 客户端addrClient*/
	sockaddr_in addrServ, addrClient;

	int len = sizeof(addrServ);

	addrServ.sin_addr.S_un.S_addr = INADDR_ANY;//自动获取IP地址INADDR_ANY
	//addrServ.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	/* INADDR_ANY表示不管是哪个网卡接收到数据，只要目的端口是SERV_PORT，就会被该应用程序接收到 */
	addrServ.sin_family = AF_INET;//使用IPV4地址
	addrServ.sin_port = htons(9998);//端口

	/*申请监听套接字*/
	SOCKET  sockServer = socket(AF_INET, SOCK_DGRAM, 0);

	if (sockServer == INVALID_SOCKET)
	{
		return WSAGetLastError();
	}
	/*绑定套接字*/
	bind(sockServer, (sockaddr*)&addrServ, len);

	while (1)
	{
		printf("server wait:\n");

		char recvBuf[1024];

		memset(recvBuf, 0, 1024);
		/*返回接收数据长度recvlen,从客户端接收到数据recvBuf，客户端地址addrClient*/
		int recvlen = recvfrom(sockServer, recvBuf, 1024, 0, (sockaddr*)&addrClient, &len);

		if (recvlen>0)
		{
			//char sIP[20];
			//inet_ntop(AF_INET, &addrClient.sin_addr, sIP, 20);
			inet_ntoa(addrClient.sin_addr);
			//client_adr = inet_ntoa(addrClient.sin_addr);
			printf("Client_address:%s Server receive:%s\n", inet_ntoa(addrClient.sin_addr), recvBuf);
			//cout << recvBuf << endl;
			//cout << sIP<<recvBuf << endl;

			/*把接收到的数据再次发送回去*/
			sendto(sockServer, recvBuf, sizeof(recvBuf), 0, (sockaddr*)&addrClient, len);
		}
	}
	/*关闭套接字和套接字库*/
	closesocket(sockServer);
	WSACleanup();

	return 0;
}
