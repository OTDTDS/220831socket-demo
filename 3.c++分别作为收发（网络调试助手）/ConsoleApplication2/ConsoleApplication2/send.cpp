//
// Created by JinxBIGBIG on 2022/8/3.
//client-�ͻ��ˣ����͵�
//����������������ӳɹ�������������ַ192.168.31.57��port=9999

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

	if (HIBYTE(wdata.wVersion) != 2 || LOBYTE(wdata.wVersion) != 2)
	{
		return -1;
	}
	/*��ʼ����ַ�ṹ�� sClient����ʵҲ�Ƿ�������ַ*/
	sockaddr_in sClient;

	sClient.sin_family = AF_INET;//ʹ��IPV4��ַ
	sClient.sin_port = htons(62513);//�˿�

	//inet_pton(AF_INET, "127.0.0.1", &sClient.sin_addr);
	sClient.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	/*��������׽���*/
	SOCKET psock = socket(AF_INET, SOCK_DGRAM, 0);

	int len = sizeof(sClient);

	char sendBuf[128];
	while (1)
	{

		memset(sendBuf, 0, sizeof(sendBuf));

		cout << "please input word:";

		cin.getline(sendBuf, 64);
		/*���ط��ͻ����ĳ���*/
		sendto(psock, sendBuf, sizeof(sendBuf), 0, (sockaddr*)&sClient, len);

	}
	return 0;
	/*�ر��׽��ֺ��׽��ֿ�*/
	//closesocket(sock);
	//WSACleanup();
}
