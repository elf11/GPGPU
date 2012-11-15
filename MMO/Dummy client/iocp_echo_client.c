/*
 * iocp-based echo client. 
 *
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <winsock2.h>
#include <mswsock.h>

#include "util.h"
#include "debug.h"
#include "sock_util.h"

#ifndef BUFSIZ
#define BUFSIZ				8192
#endif


/* client socket file handle */
static SOCKET connectfd;

enum connection_state {
	STATE_DATA_RECEIVED,
	STATE_DATA_SENT,
	STATE_CONNECTION_CLOSED
};

/* structure acting as a connection handler */
struct connection {
	SOCKET sockfd;
	char recv_buffer[BUFSIZ];
	char send_buffer[BUFSIZ];
	/* buffers used for receiving messages and then echoing them back */
	WSABUF recv_buffers[1];
	WSABUF send_buffers[1];
	size_t bytes_recv;
	size_t bytes_sent;
	WSAOVERLAPPED recv_ov;
	WSAOVERLAPPED send_ov;
};


/*
 * Initialize connection structure on given socket.
 */

static struct connection *connection_create(SOCKET sockfd)
{
	struct connection *conn = malloc(sizeof(*conn));
	//DIE(conn == NULL, "malloc");

	conn->sockfd = sockfd;
	memset(conn->recv_buffer, 0, BUFSIZ);
	memset(conn->send_buffer, 0, BUFSIZ);
	conn->recv_buffers[0].buf = conn->recv_buffer;
	conn->send_buffers[0].buf = conn->send_buffer;
	conn->recv_buffers[0].len = BUFSIZ;
	conn->send_buffers[0].len = BUFSIZ;

	memset(&conn->recv_ov, 0, sizeof(conn->recv_ov));
	memset(&conn->send_ov, 0, sizeof(conn->send_ov));

	return conn;
}


/*
 * Remove connection handler.
 */

static void connection_remove(struct connection *conn)
{
	closesocket(conn->sockfd);
	free(conn);
}



/*
 * Use WSASend to asynchronously send message through socket.
 */

static void connection_schedule_socket_send(struct connection *conn)
{
	DWORD flags;
	int rc;

	memset(&conn->send_ov, 0, sizeof(conn->send_ov));

	flags = 0;
	rc = WSASend(
			conn->sockfd,
			conn->send_buffers,
			1,
			NULL,
			flags,
			&conn->send_ov,
			NULL);
	
	if (rc != SOCKET_ERROR )
		printf("muie");
		//exit(EXIT_FAILURE);
		
	
}

/*
 * Use WSARecv to asynchronously receive message from socket.
 */

static void connection_schedule_socket_receive(struct connection *conn)
{
	DWORD flags;
	int rc;

	memset(&conn->send_ov, 0, sizeof(conn->send_ov));

	flags = 0;
	rc = WSARecv(
			conn->sockfd,
			conn->recv_buffers,
			1,
			NULL,
			&flags,
			&conn->recv_ov,
			NULL);
	if (rc != SOCKET_ERROR || WSAGetLastError() != WSA_IO_PENDING)
		;
		//exit(EXIT_FAILURE);
}





int main(int argc, char *argv[])
{
	//================De aici===============================
	static struct connection *connn;
	char buffer[256];

	wsa_init();//ai nevoie de functia asta in main
	
	connectfd = tcp_connect_to_server(argv[1], atoi(argv[2]));
	DIE(connectfd == INVALID_SOCKET, "tcp_connect_to_server");

	
	/* Instantiate new connection handler. */
	connn = connection_create(connectfd);

	//Aici in primul receive primesti datele gen pozitii initiale
	memcpy(connn->send_buffer,"-1.000000 -1.000000 -324.042542 -0.000003 -305.136108 303.976868 158.605179 -160.047546",75);
	connn->bytes_sent = 3;
	printf("Sent : %s\n",&connn->send_buffer);
	connection_schedule_socket_send(connn);
	Sleep(100);
	connection_schedule_socket_receive(connn);
	printf("buffer : %s\n",connn->recv_buffer);
	//=========================Pana aici===========================
	//=====================trebuie pus in mainul tau======================
	/* client main loop ;acest loop poate fi pus intr-o functie daca vrei*/
	while (1) {
		
		//Astepti comanda de la tastatura
		printf("Please enter the command (or 'quit' to exit): \n");
    	memset(buffer, 0 , 256);
    	fgets(buffer, 255, stdin);

		if (buffer[0]=='q'){
			tcp_close_connection(connn->sockfd);	
			exit(0);
		}

		//Copiezi comanda pentru a o trimite la server
		memcpy(connn->send_buffer,buffer,256);
		connn->bytes_sent = 256;
		printf("Sent : %s\n",&connn->send_buffer);
		connection_schedule_socket_send(connn);
		Sleep(100);
		connection_schedule_socket_receive(connn);
		//desi nu e nevoie acum in buffer e tot ce ai primit de la server
		memcpy(buffer,connn->recv_buffer,256);
		//==========================================
		//Aici poti s afolosesti campul connn->recv_buffer sau buffer pentru a procesa datele tale;eu doar le afisez
		//============================================
		printf("Received: %s\n",&connn->recv_buffer);
		
		
	}

	wsa_cleanup();//ai nevoie de functia asta in main

	return 0;
}
