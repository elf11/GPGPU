/*! \file sock_util.h
 *  \brief: useful socket macros and structures
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

#ifndef SOCK_UTIL_H_
#define SOCK_UTIL_H_	1

#ifdef __cplusplus
extern "C" {
#endif

#include <winsock2.h>

/*!\def DEFAULT_LISTEN_BACKLOG 
 * \brief default backlog for listen(2) system call 
 */
#define DEFAULT_LISTEN_BACKLOG		5

/*!\def SSA 
 * \brief "shortcut" for struct sockaddr structure 
 */
#define SSA			struct sockaddr


int wsa_init(void);
void wsa_cleanup(void);

SOCKET tcp_connect_to_server(const char *name, unsigned short port);
int tcp_close_connection(SOCKET s);
SOCKET tcp_create_listener(unsigned short port, int backlog);
int get_peer_address(SOCKET s, char *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif