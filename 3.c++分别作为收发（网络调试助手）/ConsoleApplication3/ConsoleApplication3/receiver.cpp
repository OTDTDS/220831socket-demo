//
// Created by JinxBIGBIG on 2022/8/3.
// server������-���ն�
//����������������ӳɹ����˿�ΪԶ�������Ķ˿ڣ���

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

	/*��ʼ����ַ�ṹ�� ������sa, �ͻ���recSa*/
	sockaddr_in sa, recSa;

	int len = sizeof(sa);

	sa.sin_addr.S_un.S_addr = INADDR_ANY;//�Զ���ȡIP��ַINADDR_ANY
	//sa.sin_addr.S_un.S_addr = inet_addr("192.168.31.57");
	/* INADDR_ANY��ʾ�������ĸ��������յ����ݣ�ֻҪĿ�Ķ˿���SERV_PORT���ͻᱻ��Ӧ�ó�����յ� */
	sa.sin_family = AF_INET;//ʹ��IPV4��ַ
	sa.sin_port = htons(62513);//�˿�

	/*��������׽���*/
	SOCKET  sock = socket(AF_INET, SOCK_DGRAM, 0);

	if (sock == INVALID_SOCKET)
	{
		return WSAGetLastError();
	}
	/*���׽���*/
	bind(sock, (sockaddr*)&sa, len);

	while (true)
	{
		char buf[1024];

		memset(buf, 0, 1024);
		/*���ؽ������ݳ���nlen,����buf���ͻ��˵�ַrecSa*/
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
	/*�ر��׽��ֺ��׽��ֿ�*/
	//closesocket(sock);
	//WSACleanup();
}
