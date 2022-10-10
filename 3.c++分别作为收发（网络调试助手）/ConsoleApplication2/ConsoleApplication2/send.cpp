//
// Created by JinxBIGBIG on 2022/8/3.
//client-客户端，发送的
//和网络调试助手连接成功，本地主机地址192.168.31.57，port=9999

#include <iostream>
/* inet_ntop() */
#include <WS2tcpip.h>

using namespace std;
/*链接静态库*/
#pragma comment(lib,"ws2_32.lib")

int main()
{
	/*初始化套接字库*/
	WSADATA wdata;

	WORD wVersion;

	wVersion = MAKEWORD(2, 2);

	WSAStartup(wVersion, &wdata);

	if (HIBYTE(wdata.wVersion) != 2 || LOBYTE(wdata.wVersion) != 2)
	{
		return -1;
	}
	/*初始化地址结构体 sClient，其实也是服务器地址*/
	sockaddr_in sClient;

	sClient.sin_family = AF_INET;//使用IPV4地址
	sClient.sin_port = htons(62513);//端口

	//inet_pton(AF_INET, "127.0.0.1", &sClient.sin_addr);
	sClient.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	/*申请监听套接字*/
	SOCKET psock = socket(AF_INET, SOCK_DGRAM, 0);

	int len = sizeof(sClient);

	char sendBuf[128];
	while (1)
	{

		memset(sendBuf, 0, sizeof(sendBuf));

		cout << "please input word:";

		cin.getline(sendBuf, 64);
		/*返回发送回来的长度*/
		sendto(psock, sendBuf, sizeof(sendBuf), 0, (sockaddr*)&sClient, len);

	}
	return 0;
	/*关闭套接字和套接字库*/
	//closesocket(sock);
	//WSACleanup();
}
