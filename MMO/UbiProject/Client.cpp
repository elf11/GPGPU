#include "Client.h"

Client::Client(char* host, int port)
{
  // Instantiate new connection handler.
  WSADATA wsaData;
	int iResult;
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != NO_ERROR)
		return;

  struct hostent *hent;
  hent = gethostbyname("localhost");
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if(sockfd < 0)
    printf("Error opening socket!\n");
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host, &addr.sin_addr);
  addr.sin_addr;
  int link = connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));
  if(link < 0)
    printf("Connect error\n");
  this->id = -1;
}

Client::~Client()
{
  WSACleanup();
  closesocket(sockfd);
}

void Client::sendBuffer(float* buffer, int length)
{
  char charBuffer[100];
  char aux_b[15];
  strcpy(charBuffer, "");
  for(int i = 0; i < 8; i++){
    strcpy(aux_b, "");
    sprintf(aux_b, "%f ", buffer[i]);
    strcat(charBuffer, aux_b);
  }
  send(sockfd, charBuffer, strlen(charBuffer)+1, 0);
}

float* Client::getInitial(BoundingBox *b, Vector3 &initial_poz)
{
  float* buffer = new float[8];
  memset(buffer, 0, 8*4);
  if(b->getType() ==BoundingBox::PARALLELEPIPED){
    AAParallelepiped *b2 = reinterpret_cast<AAParallelepiped*>(b);
    buffer[0] = (float)id;
    buffer[1] = (float)DEFAULT;
    buffer[2] = b2->minPoint.x + initial_poz.x;
    buffer[3] = b2->minPoint.y + initial_poz.y;
    buffer[4] = b2->minPoint.z + initial_poz.z;
    buffer[5] = b2->maxPoint.x + initial_poz.x;
    buffer[6] = b2->maxPoint.y + initial_poz.y;
    buffer[7] = b2->maxPoint.z + initial_poz.z;
  }

  return buffer;
}