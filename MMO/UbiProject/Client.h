#ifndef _CLIENT_LIB_
#define _CLIENT_LIB_

#include <ws2tcpip.h>
#include <stdio.h>
#include "CompCore.h"

#pragma comment(lib, "Ws2_32.lib")

#define BUFF_SIZE 256
/**
* \class Clasa care stabileste si se ocupa de comunicatia cu serverul
*/
class Client{
public:
  /**
  * \brief Comenzile disponibile default sunt pentru primul 
  * pachet trimis
  */
  static const int DEFAULT = -1;
  static const int MOVE = 0;
  static const int TELEPORT = 1;

  /**
  * \param id Id-ul clientului
  */
  int id;
  SOCKET sockfd;
  struct sockaddr_in addr;
  Client(char* host, int port);
  ~Client();

  /**
  * \brief Transforma in format potrivit pentru server si 
  * trimite buffer-ul apoi
  */
  void sendBuffer(float* buffer, int length);

  /**
  * \return Returneaza buffer-ul cu pozitia initiala
  * \brief diferenta dintre ce se trimite la inceput si in main
  * loop este ca aici nu se stie id-ul clientului
  */
  float* getInitial(BoundingBox *b, Vector3 &initial_poz);

  /**
  * \brief modifica id-ul clientului cu <b>id</b>
  * \param id
  */
  inline void setId(int id){this->id = id;}
};
#endif