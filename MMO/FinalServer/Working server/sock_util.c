/*!
 *\file sock_util.c
 * \brief  useful socket functions
 *
 * 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <winsock2.h>

#include "util.h"
#include "debug.h"
#include "sock_util.h"
/*!
 * Initiate use of Winsock DLL
 */
int wsa_init(void)
{
	WSADATA wsaData;
	int iResult;

	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != NO_ERROR)
		return -1;

	return 0;
}
/*!
 * Terminate the use of Winsock DLL
 */
void wsa_cleanup(void)
{
	WSACleanup();
}

/*!
 * Connect to a TCP server identified by name (DNS name or dotted decimal
 * string) and port.
 *@param name The server's DNS name/IP
 *@param port The port it listens to
 *@return A socket on which we have connected
 */

SOCKET tcp_connect_to_server(const char *name, unsigned short port)
{
	struct hostent *hent;
	struct sockaddr_in server_addr;
	SOCKET s;
	int rc;

	hent = gethostbyname(name);
	DIE(hent == NULL, "gethostbyname");

	s = socket(PF_INET, SOCK_STREAM, 0);
	DIE(s == INVALID_SOCKET, "socket");

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(port);
	memcpy(&server_addr.sin_addr.s_addr, hent->h_addr,
			sizeof(server_addr.sin_addr.s_addr));

	rc = connect(s, (struct sockaddr *) &server_addr,
				sizeof(server_addr));
	DIE(rc < 0, "connect");

	return s;
}
/*!
 * Close the connection made to a specified socket
 * @param sockfd The descriptor identifyind the socket to be closed
 * @return The result of closing the socket
 */
int tcp_close_connection(SOCKET sockfd)
{
	int rc;

	rc = shutdown(sockfd, SD_BOTH);
	DIE(rc < 0, "shutdown");

	return closesocket(sockfd);
}

/*!
 * Create a server socket.
 *@param port A port on which to listen
 *@param backlog The maximum length of the queue of pending connections
 *@return The descriptor identyfing a socket 
 */

SOCKET tcp_create_listener(unsigned short port, int backlog)
{
	struct sockaddr_in address;
	SOCKET listenfd;
	int sock_opt;
	int rc;

	listenfd = socket(PF_INET, SOCK_STREAM, 0);
	DIE(listenfd == INVALID_SOCKET, "socket");

	sock_opt = 1;
	rc = setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
				(const char *) &sock_opt, sizeof(int));
	DIE(rc < 0, "setsockopt");

	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons(port);
	address.sin_addr.s_addr = INADDR_ANY;

	rc = bind(listenfd, (SSA *) &address, sizeof(address));
	DIE(rc < 0, "bind");

	rc = listen(listenfd, backlog);
	DIE(rc < 0, "listen");

	return listenfd;
}

/*!
 * Use getpeername(2) to extract remote peer address. 
 *@param s A descriptor identifying a socket
 *@param buf Fill buffer with address format IP_address:port (e.g. 192.168.0.1:22).
 *@param len Size of buffer
 *@return A value that is 0 for succes and -1 for failure
 */

int get_peer_address(SOCKET s, char *buf, size_t len)
{
	struct sockaddr_in addr;
	size_t addrlen = sizeof(struct sockaddr_in);

	if (getpeername(s, (SSA *) &addr, &addrlen) < 0)
		return -1;

	sprintf_s(buf, len, "%s:%d", inet_ntoa(addr.sin_addr), ntohs(addr.sin_port));

	return 0;
}
