//
// Created by JinxBIGBIG on 2022/8/3.
// server������

#include <iostream>
#include <WinSock2.h>
/* inet_ntop() */
#include <WS2tcpip.h>
/*���Ӿ�̬��*/
#pragma comment (lib,"ws2_32.lib")
using namespace  std;

int main()
{    /*��ʼ���׽��ֿ�*/
	WSADATA  wsData;

	int nret = WSAStartup(MAKEWORD(2, 2), &wsData);
	if (nret != 0)
	{
		return nret;
	}

	/*��ʼ����ַ�ṹ�� ������addrServ, �ͻ���addrClient*/
	sockaddr_in addrServ, addrClient;

	int len = sizeof(addrServ);

	addrServ.sin_addr.S_un.S_addr = INADDR_ANY;//�Զ���ȡIP��ַINADDR_ANY
	//addrServ.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	/* INADDR_ANY��ʾ�������ĸ��������յ����ݣ�ֻҪĿ�Ķ˿���SERV_PORT���ͻᱻ��Ӧ�ó�����յ� */
	addrServ.sin_family = AF_INET;//ʹ��IPV4��ַ
	addrServ.sin_port = htons(9998);//�˿�

	/*��������׽���*/
	SOCKET  sockServer = socket(AF_INET, SOCK_DGRAM, 0);

	if (sockServer == INVALID_SOCKET)
	{
		return WSAGetLastError();
	}
	/*���׽���*/
	bind(sockServer, (sockaddr*)&addrServ, len);

	while (1)
	{
		printf("server wait:\n");

		char recvBuf[1024];

		memset(recvBuf, 0, 1024);
		/*���ؽ������ݳ���recvlen,�ӿͻ��˽��յ�����recvBuf���ͻ��˵�ַaddrClient*/
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

			/*�ѽ��յ��������ٴη��ͻ�ȥ*/
			sendto(sockServer, recvBuf, sizeof(recvBuf), 0, (sockaddr*)&addrClient, len);
		}
	}
	/*�ر��׽��ֺ��׽��ֿ�*/
	closesocket(sockServer);
	WSACleanup();

	return 0;
}
