//
// Created by JinxBIGBIG on 2022/8/3.
//client-�ͻ���

#include <iostream>
/* inet_ntop() */
#include <WS2tcpip.h>

using namespace std;
/*���Ӿ�̬��*/
#pragma comment(lib,"ws2_32.lib")

int main()
{
	/*��ʼ���׽��ֿ�*/
	WSADATA wdata;

	WORD wVersion;

	wVersion = MAKEWORD(2, 2);

	WSAStartup(wVersion, &wdata);
	
	//int nret = WSAStartup(MAKEWORD(2, 2), &wsData);
	//if (nret != 0)
	//{
	//	return nret;
	//}

	if (HIBYTE(wdata.wVersion) != 2 || LOBYTE(wdata.wVersion) != 2)
	{
		return -1;
	}

	/*��ʼ����ַ�ṹ�� addrServ����ʵҲ�Ƿ�������ַ*/
	sockaddr_in addrServ;

	int len = sizeof(addrServ);
	//inet_pton(AF_INET, "127.0.0.1", &addrServ.sin_addr);
	addrServ.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	addrServ.sin_family = AF_INET;//ʹ��IPV4��ַ
	addrServ.sin_port = htons(9998);//�˿�



	/*��������׽���*/
	SOCKET sockClient = socket(AF_INET, SOCK_DGRAM, 0);


	while (1)
	{
		char sendBuf[128];
		memset(sendBuf, 0, sizeof(sendBuf));

		cout << "pelase input word:";

		cin.getline(sendBuf, 64);
		/*���ط��ͻ����ĳ���*/
		sendto(sockClient, sendBuf, sizeof(sendBuf), 0, (sockaddr*)&addrServ, len);

		/*���շ��ػ���������*/

		char recvBuf[128];
		memset(recvBuf, 0, 128);
		int recvlen = recvfrom(sockClient, recvBuf, 128, 0, (sockaddr*)&addrServ, &len);

		printf("Client receive:%s\n", recvBuf);//��ӡ���������



	}

	/*�ر��׽��ֺ��׽��ֿ�*/
	closesocket(sockClient);
	WSACleanup();

	return 0;
}
