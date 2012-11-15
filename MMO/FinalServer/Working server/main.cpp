
/*! \file main.cpp
 *  \brief iocp-based server modified. Uses IoCompletionPorts to wait for
 * multiple operations.

 * 
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <winsock2.h>
#include <mswsock.h>

#include <math.h>
#include <stdarg.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <conio.h>

#include "util.h"
#include "debug.h"
#include "sock_util.h"
#include "w_iocp.h"
/*! \def ECHO_LISTEN_PORT
 *  \brief The port the server is listening on
 */
#define ECHO_LISTEN_PORT		42425
/*! \def BUFSIZ
 *  \brief Size of the recv/send buffer of the connection handler
 */
#ifndef BUFSIZ
#define BUFSIZ				8192
#endif
/*! \def DECODED_SIZE
 *  \brief Size of the recv/send buffer of the decoded message
 *  \see decodeEncodeInfo()
 */
#ifndef DECODED_SIZE
#define DECODED_SIZE		8
#endif

/*! \var SOCKET listenfd
 *  \brief server socket file handle 
 */
static SOCKET listenfd;

/*! \var HANDLE iocp
 *  \brief IoCompletionPort handle 
 */
static HANDLE iocp;

enum connection_state {
	STATE_DATA_RECEIVED,
	STATE_DATA_SENT,
	STATE_CONNECTION_CLOSED
};

/*!
 * structure acting as a connection handler
 * @param sockfd A socket bound to the handler
 * @param recv_buffer Receiving buffer
 * @param send_buffer Sending buffer
 * @param bytes_recv Size of receiving buffer
 * @param bytes_send Size of sending buffer 
 */
struct connection {
	SOCKET sockfd;
	char recv_buffer[BUFSIZ];
	char send_buffer[BUFSIZ];
	/* buffers used for receiving messages and then echoing them back */
	WSABUF recv_buffers[1];
	WSABUF send_buffers[1];
	DWORD bytes_recv;
	DWORD bytes_sent;
	WSAOVERLAPPED recv_ov;
	WSAOVERLAPPED send_ov;
};

/*!
 * Anonymous structure used to "confine" data regardin asynchronous accept
 * operations (handled through AcceptEx and Io Completion Ports).
 */
static struct {
	SOCKET sockfd;
	char buffer[BUFSIZ];
	DWORD len;
	OVERLAPPED ov;
} ac;

/*
 * Data for encoding/decoding the information
 */
 /*! \var int TYPE
  *  \brief Type of movement : 0 - walk ,1 - teleport
  */
int TYPE;
 /*! \var int ID
  *  \brief client ID - Initially -1
  */
int ID = -1;
 /*! \var int maxId
  *  \brief Number of connected clients
  */
int maxId = 0;






////////////////////////////////////////////
/*! \def min (x,y)
 *  \brief Determines the minimum between two numbers
 */
#define min(x, y) (x < y ? x : y)
/*! \def ABS(x)
 *  \brief Determines the absolute value of a number
 */
#define ABS(x) (x < 0 ? -x : x)

 /*! \var float * BBstaticCenter_d
  *  \brief List of all the centers of the bounding boxes of the static objects, used on the GPU
  */
float * BBstaticCenter_d;
 /*! \var float * BBdinamicCenter_d
  *  \brief List of all the centers of the bounding boxes of the connected clients, used on the GPU
  */ 
float * BBdinamicCenter_d;
 /*! \var float * BBstaticSize_d
  *  \brief List of all the sizes of the bounding boxes of the static objects, used on the GPU
  */
float * BBstaticSize_d;
 /*! \var float * BBdinamicSize_d
  *  \brief List of all the sizes of the bounding boxes of the connected clients used on the GPU
  */ 
float * BBdinamicSize_d;
 /*! \var float * BBstaticCenter
  *  \brief List of all the centers of the bounding boxes of the static objects, used on the CPU
  *  \var float * BBdinamicCenter
  *  \brief List of all the centers of the bounding boxes of the connected clients, used on the CPU
  */ 
float * BBstaticCenter, * BBdinamicCenter;
 /*! \var float * BBstatic
  *  \brief List of the positions of the bounding boxes of the static objects
  */ 
float * BBstatic = NULL;
 /*! \var float * BBdinamic
  *  \brief List of the positions of the bounding boxes of the connected clients
  */ 
float * BBdinamic;
/*!  \var float * BBstaticSize
  *  \brief List of all the sizes of the bounding boxes of the static objects, used on the CPU
  *  \var float * BBdinamicSize
  *  \brief List of all the sizes of the bounding boxes of the connected clients used on the CPU
  */
float * BBstaticSize, * BBdinamicSize;
/*!  \var int  nrBBstatic
  *  \brief Number of  the static objects
  *  \var int  nrBBdinamic
  *  \brief Number of all the connected clients
  */
int nrBBstatic = 0, nrBBdinamic = 0;
/*!  \var int * okStatic_d
  *  \brief Records if a there was a hit with the static objecs,used on GPU
  *  \var int * okDinamic_d
  *  \brief Records if a there was a hit with the connected clients,used on GPU
  */
int *okStatic_d , *okDinamic_d;
/*!  \var int * okStatic
  *  \brief Records if a there was a hit with the static objecs,used on CPU
  *  \var int * okDinamic
  *  \brief Records if a there was a hit with the connected clients,used on CPU
  */
int *okStatic, *okDinamic;

/*! \brief Functia de lansare a kernelului CUDA
 * Performs collision tests between the connected client and all the static objects in the scene
 * Each GPU thread tests a collision with only one static object 
 *@param BBstaticCenter_d A list with all the centers of the static objects
 *@param nrBBstatic Number of  the static objects
 *@param BBstaticSize_d List of all the sizes of the bounding boxes of the static objects
 *@param xFuture x coordinate of the new position
 *@param yFuture y coordinate of the new position
 *@param zFuture z coordinate of the new position
 *@param size1 The size of the bounding box of the client on the x dimension
 *@param size2 The size of the bounding box of the client on the z dimension
 *@param okStatic The recorded/or not hit
 *@param DIM_GRID Has a value of (1,1,1)
 *@param DIM_BLOCK Has a value of (nrBBstatic,1,1)
 *\see BBstaticCenter_d 
 *\see nrBBstatic , BBstaticSize_d,okStatic
 */
extern "C"
	cudaError_t launch_CubeStatic(float * BBstaticCenter_d,int nrBBstatic, float * BBstaticSize_d, 
								  float xFuture, float yFuture, float zFuture, float size1, float size2, 
								  int *okStatic,dim3 DIM_GRID,dim3 DIM_BLOCK);

/*! \brief Functia de lansare a kernelului CUDA
 * Performs collision tests between the connected client and all the other clients in the scene
 * Each GPU thread tests a collision with only one client 
 *@param BBdinamicCenter_d A list with all the centers of the connected clients
 *@param nrBBdinamic Number of  the connected clients
 *@param BBdinamicSize_d List of all the sizes of the bounding boxes of the connected clients
 *@param xFuture x coordinate of the new position
 *@param yFuture y coordinate of the new position
 *@param zFuture z coordinate of the new position
 *@param ID The client id
 *@param okDinamic The recorded/or not hit
 *@param DIM_GRID Has a value of (1,1,1)
 *@param DIM_BLOCK Has a value of (nrBBdinamic,1,1)
 *\see BBdinamicCenter_d 
 *\see nrBBdinamic , BBdinamicSize_d,okDinamic
 */
extern "C" 
	cudaError_t launch_CubeDinamic(float * BBdinamicCenter_d, int nrBBdinamic, float * BBdinamicSize_d, 
							       float xFuture, float yFuture, float zFuture, int ID, int *okDinamic,
								   dim3 DIM_GRID,dim3 DIM_BLOCK);



unsigned int timer = 0;


char * decodeEncodeInfo(char info[]);
int CudaWork(int ID, int x, int y, int z);
void WriteAtPoz(char *filename,int poz,DWORD type,char* buffer);

/*!
 * Initialize connection structure on given socket.
 *@param sockfd The given socket
 *@return A "connection" handler bound to that socket
 *\see connection 
 */

static struct connection *connection_create(SOCKET sockfd)
{
	struct connection *conn = (struct connection*)malloc(sizeof(*conn));
	DIE(conn == NULL, "malloc");

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

/*!
 * Add a non bound socket to the connection. The socket is to be bound
 * by AcceptEx.
 *@return A boundless "connection" handler
 *\see connection
 *\see create_iocp_accept()
 */

static struct connection *connection_create_with_socket(void)
{
	SOCKET s;

	s = socket(PF_INET, SOCK_STREAM, 0);
	DIE(s == INVALID_SOCKET, "socket");

	return connection_create(s);
}

/*!
 * Remove connection handler.
 *@param conn The connection handler
 */

static void connection_remove(struct connection *conn)
{
	dlog(LOG_LEVEL, "Remove connection.\n");
	closesocket(conn->sockfd);
	free(conn);
}

/*!
 * Prepare data for overlapped I/O send operation.
 * Decode th information received and encode the response after processing
 * @param conn A connection handler
 * \see connection
 * \see decodeEncodeInfo()
 */

static void connection_prepare_socket_send(struct connection *conn)
{
	
		char * result = decodeEncodeInfo(conn->recv_buffer);
	memcpy(conn->send_buffer,result ,strlen(result)+1);
	conn->send_buffers[0].len = strlen(result)+1;
	

}

/*!
 * Use WSASend to asynchronously send message through socket.
 * @param conn The connection handler
 *\see connection
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
	
	
}

/*!
 * Use WSARecv to asynchronously receive message from socket.
 * @param conn The connection handler
 *\see connection
 */

static void connection_schedule_socket_receive(struct connection *conn)
{
	DWORD flags;
	int rc;
	//printf("Se apeleaza!!!\n");
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
	
}

/*!
 * Overllaped I/O send operation completed (as signaled by I/O Completion
 * Port).
 * After completion start receiving again
 * @param conn The connection handler
 * @param ovp Overllaped I/O info
 * \see connection 
 * \see connection_schedule_socket_receive()
 */

static void connection_complete_socket_send(struct connection *conn, WSAOVERLAPPED *ovp)
{
	connection_schedule_socket_receive(conn);
}

/*!
 * Overllaped I/O receive operation completed (as signaled by I/O Completion
 * Port). Send message back after processing.
 * @param conn The connection handler
 * @param ovp Overllaped I/O info
 * \see connection
 * \see connection_prepare_socket_send(),connection_schedule_socket_send()
 */

static void connection_complete_socket_receive(struct connection *conn, WSAOVERLAPPED *ovp)
{
	BOOL bRet;
	DWORD flags;

	bRet = WSAGetOverlappedResult(
			conn->sockfd,
			ovp,
			&conn->bytes_recv,
			FALSE,
			&flags);
	DIE(bRet == FALSE, "WSAGetOverlappedResult");

	/* In case of no bytes received, consider connection terminated. */
	if (conn->bytes_recv == 0) { 
		connection_remove(conn);
		
		return;
	}
	/*!Now prepare for sending*/
	connection_prepare_socket_send(conn);
	/*! And send */
	connection_schedule_socket_send(conn);
	
}

/*!
 * Schedule overlapped operation for accepting a new connection.
 */

static void create_iocp_accept(void)
{
	BOOL bRet;

	memset(&ac, 0, sizeof(ac));

	/* Create simple socket for acceptance */
	ac.sockfd = socket(PF_INET, SOCK_STREAM, 0);
	DIE(ac.sockfd == INVALID_SOCKET, "socket");

	/* Launch overlapped connection accept through AcceptEx. */
	bRet = AcceptEx(
			listenfd,
			ac.sockfd,
			ac.buffer,
			0,
			128,
			128,
			&ac.len,
			&ac.ov);
	DIE(bRet == FALSE && WSAGetLastError() != ERROR_IO_PENDING, "AcceptEx");
}

/*!
 * Handle a new connection request on the server socket.
 * \see connection_create(),connection_schedule_socket_receive(),create_iocp_accept()
 */

static void handle_new_connection(OVERLAPPED *ovp)
{
	struct connection *conn;
	char abuffer[64];
	HANDLE hRet;
	int rc;

	rc = setsockopt(
			ac.sockfd,
			SOL_SOCKET,
			SO_UPDATE_ACCEPT_CONTEXT,
			(char *) &listenfd,
			sizeof(listenfd)
		  );
	DIE(rc < 0, "setsockopt");

	rc = get_peer_address(ac.sockfd, abuffer, 64);
	if (rc < 0) {
		ERR("get_peer_address");
		return;
	}

	dlog(LOG_DEBUG, "Accepted connection from %s\n", abuffer);

	/*! Instantiate new connection handler. */
	conn = connection_create(ac.sockfd);

	/*! Add socket to IoCompletionPort. */
	hRet = w_iocp_add_key(iocp, (HANDLE) conn->sockfd, (ULONG_PTR) conn);
	DIE(hRet != iocp, "w_iocp_add_key");

	/*! Schedule receive operation. */
	connection_schedule_socket_receive(conn);
	/* We have a new client */
	ID = maxId;
	maxId++;
	
	

	
	/*! Use AcceptEx to schedule new connection acceptance. */
	create_iocp_accept();

}

/*!
 * Process overlapped I/O operation: data has been received from or
 * has been sent to the socket.
 * @param conn The connection handler
 * @param bytes 
 * @param ovp Overllaped I/O info
 * \see connection_complete_socket_send(),connection_complete_socket_receive()
 */

static void handle_aio(struct connection *conn, size_t bytes, OVERLAPPED *ovp)
{
	if (ovp == &conn->send_ov) {
		//printf("deci asta e de send....\n");
		dlog(LOG_LEVEL, "Send operation completed.\n");
		connection_complete_socket_send(conn, ovp);
	}
	else if (ovp == &conn->recv_ov) {
		//printf("deci aici intra\n");
		dlog(LOG_LEVEL, "Recv operation completed.\n");
		connection_complete_socket_receive(conn, ovp);
	}
	else
		dlog(LOG_ERR, "Unknown operation completed.\n");
}
/*!\brief Function that decodes the information recceived and encodes the information after processing it
 * @param info The information received
 * @return The information after processing it
 * \see WriteAtPoz(),CudaWork()
 */
char * decodeEncodeInfo(char info[])
{
	float  decoded[DECODED_SIZE];
	char * pch,  * encoded = (char*)calloc(maxId*6+1,sizeof(float)) , extension[200];
	char aux[200];
	int i = 0 ,j;
	int hitAndModify;
	
	//printf("Received :::::: %s\n",info);
	strcpy(extension,info);
	/*! We decode the information received */
	pch = strtok(info," ");
	while (pch != NULL) {
		decoded[i++] = (float) atoi(pch);
		//printf("Valorile noastre:%f ",decoded[i-1]);
		pch = strtok (NULL, " ");
	
	}
	/*! If we have a new client then We need to write its position to the "dinamic_bounding_box.bb" file */
	if(decoded[0] == -1){
		//printf("asta e nr de clienti : %d\n",maxId);
		sprintf(aux,"%d",maxId);
		strcat(aux,"\n");
		WriteAtPoz("dinamic_bounding_box.bb",0,FILE_BEGIN,aux);

		WriteAtPoz("dinamic_bounding_box.bb",0,FILE_END,"\n0\n");
		//printf("Coordonate:%s\n",extension);
		WriteAtPoz("dinamic_bounding_box.bb",0,FILE_END,extension+20);
		decoded[0] = ID;
	}
	


	/*! After decoding "decoded" contains ID,type,client bounding box */
	/*! Start processing the information -CudaWork- */
	//???????????????????????????????????????????????????????
	hitAndModify = CudaWork(decoded[0],decoded[2]+(decoded[2]+decoded[5])/2,decoded[3]+(decoded[3]+decoded[6])/2,decoded[4]+(decoded[4]+decoded[7])/2);
	if(hitAndModify == 0){
	/*! After processing the information we need to update -if necessary- our client position from the file */
	//printf("Writing back for future generations\n");
	HANDLE hWrite = CreateFile(
						 "dinamic_bounding_box.bb",
						 GENERIC_WRITE,	   /* access mode */
						 FILE_SHARE_WRITE,	   /* sharing option */
						 NULL,		   /* security attributes */
						 OPEN_EXISTING,	   /* open only if it exists */
						 FILE_ATTRIBUTE_NORMAL,/* file attributes */
						 NULL
					);
	SetFilePointer(hWrite,0,NULL,FILE_BEGIN);
	SetEndOfFile(hWrite);
	CloseHandle(hWrite);
	sprintf(aux,"%d",maxId);
	strcat(aux,"\n");
	
	WriteAtPoz("dinamic_bounding_box.bb",0,FILE_BEGIN,aux);

	WriteAtPoz("dinamic_bounding_box.bb",0,FILE_END,"\n0\n");
	
	
	printf("Coordonate extension:%s\n",extension);
	WriteAtPoz("dinamic_bounding_box.bb",0,FILE_END,extension+20);
	
	}
	
	/*! Now we have to encode the ID , current client position and all the other clients that are still connected*/
	sprintf(aux,"%d",decoded[0]);
	strcpy(encoded,aux);
	strcat(encoded," ");
	sprintf(aux,"%d ",maxId);
	strcat(encoded,aux);
	sprintf(aux,"%f",BBdinamic[(int)decoded[0]]);
	strcat(encoded,aux);
	strcat(encoded," ");
	sprintf(aux,"%f",BBdinamic[(int)decoded[0]+1]);
	strcat(encoded,aux);
	strcat(encoded," ");
	sprintf(aux,"%f",BBdinamic[(int)decoded[0]+2]);
	strcat(encoded,aux);
	strcat(encoded," ");
	sprintf(aux,"%f",BBdinamic[(int)decoded[0]+3]);
	strcat(encoded,aux);
	strcat(encoded," ");
	sprintf(aux,"%f",BBdinamic[(int)decoded[0]+4]);
	strcat(encoded,aux);
	strcat(encoded," ");
	sprintf(aux,"%f",BBdinamic[(int)decoded[0]+5]);
	strcat(encoded,aux);
	strcat(encoded," ");
	//for 3D
	//sprintf(aux,"%g",retunPoz[decoded[0]+2]);
	//strcat(encoded,aux);
	for (j = 0 ; j < maxId; j++) {
		if(j!=decoded[0]){
			sprintf(aux,"%f",BBdinamic[j]);
			strcat(encoded,aux);
			strcat(encoded," ");
			sprintf(aux,"%f",BBdinamic[j+1]);
			strcat(encoded,aux);
			strcat(encoded," ");
			sprintf(aux,"%f",BBdinamic[j+2]);
			strcat(encoded,aux);
			strcat(encoded," ");
			sprintf(aux,"%f",BBdinamic[j+3]);
			strcat(encoded,aux);
			strcat(encoded," ");
			sprintf(aux,"%f",BBdinamic[j+4]);
			strcat(encoded,aux);
			strcat(encoded," ");
			sprintf(aux,"%f",BBdinamic[j+5]);
			strcat(encoded,aux);
			strcat(encoded," ");
			
		}
	}
	strcpy(extension,"");
	strcat(encoded,"\0");
	//printf("Encoding Succeded: %s\n",encoded);
	//free(BBdinamic);
	//free(BBstatic);
	//free(BBstaticCenter);
	//free(BBstaticSize);
	//free(BBdinamicSize);
		return encoded;
}
/*!
 * Function that writes at the specified position in a file
 * @param filename The file where the information will be written
 * @param poz The offset at which we write the information
 * @param type The starting position - Can only be : FILE_BEGIN,FILE_END,FILE_CURRENT
 * @param buffer The information to be added to the file
 * \warning The file must exist beforehand for this to be successful
 */
void WriteAtPoz(char *filename,int poz,DWORD type,char* buffer)
{	BOOL bRet;
	DWORD WBytes,Poz;
	HANDLE hWrite = CreateFile(
							 filename,
							 GENERIC_WRITE,	   /* access mode */
							 FILE_SHARE_WRITE,	   /* sharing option */
							 NULL,		   /* security attributes */
							 OPEN_EXISTING,	   /* open only if it exists */
							 FILE_ATTRIBUTE_NORMAL,/* file attributes */
							 NULL
						);
	//printf("Opened File:%s,%d,%d,%s\n",filename,poz,type,buffer);
	Poz = SetFilePointer(
						 hWrite,
						 poz,
						 NULL,
						 type
				);

	bRet = WriteFile(  hWrite,          /* open file handle */
					   buffer,       /* start of data to write */
					   strlen(buffer), /* number of bytes to write */
					   &WBytes,/* number of bytes that were written */
					   NULL            /* no overlapped structure */
					);
	CloseHandle(hWrite);

}
/*!
 * Function that reads the number of static objects or connected clietns from the specified file
 * @param filename The file from where the reading is performed
 * @return the number of either static objects or connected clients.
 * \see nrBBStatic,nrBBdinamic
 */
int ReadNrBB(char * filename)
{
	FILE * f = fopen(filename, "r");
	if (!f)
	{
		printf("bla\n");
	}
	int nrBB = 0;
	if (f)
	{
		fscanf (f, "%d", &nrBB);
	}
	fclose(f);

	return nrBB;
}
/*!
 * Function that reads the bounding boxes of static objects or connected clietns from the specified file
 * @param filename The file from where the reading is performed
 * @param nrBB The number of either static objects or connected clients
 * @return The bounding boxes of either static objects or connected clients.
 * \see nrBBStatic,nrBBdinamic
 * \see BBdinamic,BBstatic
 */
float * ReadBoundingBox(char * filename, int nrBB)
{
	FILE * f = fopen(filename, "r");

	
	if (!f)
	{
		printf("bla\n");
	}

	float * BBArray = NULL;
	if (f)
	{
		int var;
		fscanf (f, "%d", &var);
		//printf("Vrem sa alocam azi\n");
		
		float * Secondary = (float *)calloc(nrBB * 6, sizeof(float));
		
		BBArray = Secondary;
		//printf("Am alocat\n");
		int j = 0;
		for (int i = 0; i < nrBB; i += 1)
		{
			int type;
			fscanf (f, "%d", &type);
			if (type == 0)
			{
				float minX, minY, minZ, maxX, maxY, maxZ;
				fscanf (f, "%f %f %f %f %f %f", &minX, &minY, &minZ, &maxX, &maxY, &maxZ);
				Secondary[j] = minX;
				Secondary[j + 1] = minY;
				Secondary[j + 2] = minZ;
				Secondary[j + 3] = maxX;
				Secondary[j + 4] = maxY;
				Secondary[j + 5] = maxZ;
				j +=  6;
			}
		}
	}
	//printf("First Read ready\n");

	//for (int i = 0; i < nrBB; i += 1)
	//{
	//	printf("%f\n", BBArray[i]);
	//}
	fclose(f);
	return BBArray;
}
/*!
 * Function that calculates the sizes of the bounding boxes . 
 * The bounding boxes can be either clients or static objects
 * @param BBArray The bounding boxes given as coordinates
 * @param nrBB Number of bounding boxes
 * @return An array containing the sizes of the bounding boxes
 * \see BBstatic,BBdinamic,nrBBstatic,nrBBdinamic
 * \see BBstaticSize,BBdinamicSize
 */
float * CalculateSize(float *BBArray, int nrBB)
{
	int j = 0;
	float * SizeArray = (float*)calloc( nrBB * 2,sizeof(float));

	for (int i = 0; i < nrBB * 6; i += 6)
	{
		

		float x ;
		x = abs(BBArray[i] - BBArray[i+3]);
		SizeArray[j] = x;
		j += 1;
		x = abs(BBArray[i+2] - BBArray[i+5]);
		SizeArray[j] = x;
		j += 1;
	}

	return SizeArray;
}

/*!
 * Function that calculates the centers of the bounding boxes . 
 * The bounding boxes can be either clients or static objects
 * @param BBArray The bounding boxes given as coordinates
 * @param nrBB Number of bounding boxes
 * @return An array containing the sizes of the bounding boxes
 * \see BBstatic,BBdinamic,nrBBstatic,nrBBdinamic
 * \see BBstaticCenter,BBdinamicCenter
 */
float * CalculateCenter(float * BBArray, int nrBB)
{
	int j = 0;
	float * CenterArray = (float *) calloc(nrBB * 2,sizeof(float));

	for (int i = 0; i < nrBB * 6; i += 6)
	{
		float x,y;
		x = min(BBArray[i], BBArray[i+3]);
		float u,o;
		u = BBArray[i+3];
		o = BBArray[i];
		u -= o;
		y = abs(u);
		CenterArray[j] =  x + y;
		j += 1;
		x = min(BBArray[i+2], BBArray[i+5]);
		u = BBArray[i+2];
		o = BBArray[i+5];
		u -= o;
		y = abs(u);
		CenterArray[j] = x + y;
		j += 1;
	}
	
	return CenterArray;
}
/*!
 * Function that initiates all the data it requires from the files containing both the static objects and the connected clients.
 * It also allocates all the memory it needs for performing calculus on tthe GPU.
 * @param filenameStatic The file where the bounding boxes of the static objects are
 * @param filenameDinamic  The file where the bounding boxes of the connected clients are
 * \see BBstatic, BBdinamic, nrBBstatic, nrBBdinamic
 * \see BBstaticCenter, BBdinamicCenter, BBstaticSize, BBdinamicSize 
 */
void init(char * filenameStatic, char * filenameDinamic)
{
	nrBBstatic = ReadNrBB(filenameStatic);
	nrBBdinamic = ReadNrBB(filenameDinamic);

	BBstatic = (float *)calloc(nrBBstatic * 6, sizeof(float *));
	BBdinamic = (float *)calloc(nrBBdinamic * 6, sizeof(float *));
	BBstaticSize = (float *)calloc(nrBBstatic * 2, sizeof(float *));
	BBdinamicSize = (float *)calloc(nrBBdinamic * 2, sizeof(float *));
	BBstaticCenter = (float *)calloc(nrBBstatic * 2, sizeof(float *));
	BBdinamicCenter = (float *)calloc(nrBBdinamic * 2, sizeof(float *));
	okStatic = (int *)calloc(1, sizeof(int*));
	okDinamic = (int *)calloc(1, sizeof(int*));


	//printf("Reading from files\n");
	BBstatic = ReadBoundingBox(filenameStatic, nrBBstatic);
	BBdinamic = ReadBoundingBox(filenameDinamic, nrBBdinamic);
	//for (int i = 0; i < nrBBdinamic * 6; i+=1)
	//{
	//	printf("BBdinamic from files %i : %f\n", i, BBdinamic[i]);
	//}
	//printf("Doing the calculus\n");
	BBstaticSize = CalculateSize(BBstatic, nrBBstatic);
	//printf("After the first size cube\n");
	BBdinamicSize = CalculateSize(BBdinamic, nrBBdinamic);
	//printf("After the second size cube\n");
	BBstaticCenter = CalculateCenter(BBstatic, nrBBstatic);
	BBdinamicCenter = CalculateCenter(BBdinamic, nrBBdinamic);
	//printf("Chestii CUDA\n");

	// Aloca memorie - CUDA
	cutilSafeCall(cudaMalloc((void **) &BBstaticCenter_d, (nrBBstatic *2)*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &BBdinamicCenter_d, (nrBBdinamic * 2)*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &BBstaticSize_d, (nrBBstatic * 2)*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &BBdinamicSize_d, (nrBBdinamic * 2)*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **) &okStatic_d, sizeof(int)));
	cutilSafeCall(cudaMalloc((void **) &okDinamic_d, sizeof(int)));
}

/*!Function that frees the memory used for GPU computation*/
void cleanup()
{
	//free(BBdinamic);
	//free(BBstatic);
	//free(BBstaticCenter);
	//free(BBstaticSize);
	//free(BBdinamicSize);
	cutilCheckError(cutStopTimer(timer));
	cutilCheckError(cutDeleteTimer( timer));

	cudaFree(BBdinamicCenter_d);
	cudaFree(BBstaticCenter_d);
	cudaFree(BBstaticSize_d);
	cudaFree(BBdinamicSize_d);
	cudaFree(okStatic_d);
	cudaFree(okDinamic_d);
	cudaThreadExit();
      
}

/*! Verifica daca exista eroare CUDA*/
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();

    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,  cudaGetErrorString( err) );
		getchar();
        exit(EXIT_FAILURE);
    }                         
}
/*! Initialize CUDA*/
bool initCUDA(void)
{
#if __DEVICE_EMULATION__
	return true;
#else
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "Nu exista nici un device.\n");
		return false;
	}

	printf("Exista %d device-uri.\n",count);
	
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "Nu exista nici un device care suporta CUDA.\n");
		return false;
	}
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	
	printf("CUDA initializat cu succes\n");

	// Create the CUTIL timer
    cutilCheckError( cutCreateTimer( &timer));

	return true;
#endif
}

/*! Afisare text */
void printHeader(char *s)
{
	int line_len = 79;
	line_len -= strlen(s);
	line_len /=2;

	for(int i=0;i<line_len;i++)
		printf("*");
	printf("%s",s);
	for(int i=0;i<line_len;i++)
		printf("*");
	printf("\n");
}
/*! Function that is responsible with performing computation in order to see if a collision has been detected.
 * @param Id The client id
 * @param xFuture The position at which the client wants to move on the X axis
 * @param yFuture The position at which the client wants to move on the Y axis
 * @param zFuture The position at which the client wants to move on the Z axis
 * @return It return 1 if the client has hit something, 0 otherwise
 */
int CudaWork(int Id, int xFuture, int yFuture, int zFuture)
{	
	int hit = 0;
	printHeader("Initializare");
	//printf("--------------------------------------%d-----------------------\n",Id);
	/** Initialize CUDA*/
	initCUDA();
	
	//if (BBstatic == NULL)
	//{
	/** Initialize our data from files*/
		init("static_bounding_box.bb", "dinamic_bounding_box.bb");
//	}
	//printf("Result before\n");
	//	for (int i = 0; i < nrBBdinamic*2; i += 1)
	//{
	//	printf("%f\n", BBdinamicCenter[i]);
	//}
	//printf("Done Init\n");

	//Copy the static arrays to device
	cutilSafeCall(cudaMemcpy(BBstaticCenter_d, BBstaticCenter,(nrBBstatic*2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");
	cutilSafeCall(cudaMemcpy(BBstaticSize_d, BBstaticSize,(nrBBstatic*2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	cutilSafeCall(cudaMemcpy(okStatic_d, okStatic,sizeof(int),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");
	cutilSafeCall(cudaMemcpy(okDinamic_d, okDinamic,sizeof(int),cudaMemcpyHostToDevice));
	checkCUDAError("cudaMemcpy");

	// Run Kernel for static objects
	

	float size1 = BBdinamicSize[Id];
	float size2 = BBdinamicSize[Id+1];

	int block_size = nrBBstatic;
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGridCubes( 1, 1, 1);
 /** Check collision with static objects*/
	cutilSafeCall(launch_CubeStatic(BBstaticCenter_d,(nrBBstatic*2),BBstaticSize_d,
									xFuture,yFuture,zFuture,size1,size2,okStatic_d,dimGridCubes,dimBlock));
	

	cutilSafeCall(cudaThreadSynchronize());
	cutilSafeCall(cudaMemcpy(okStatic, okStatic_d, sizeof(int), cudaMemcpyDeviceToHost));
	checkCUDAError("invocare kernel");

	// If the object we are testing is not in conflict with any of the static objects then we can move on and test with the dynamic objects
	//otherwise we just keep the result we already have.
	if (*okStatic == 0)
	{
		//Copy the dynamic arrays to the device
		cutilSafeCall(cudaMemcpy(BBdinamicCenter_d, BBdinamicCenter,(nrBBdinamic*2)*sizeof(float),cudaMemcpyHostToDevice));
		checkCUDAError("cudaMemcpy");

		cutilSafeCall(cudaMemcpy(BBdinamicSize_d, BBdinamicSize,(nrBBdinamic*2)*sizeof(float),cudaMemcpyHostToDevice));
		checkCUDAError("cudaMemcpy");

		
		//Run kernel for dynamic objects.
		int block_size = nrBBdinamic;
	    dim3 dimBlock(block_size, 1, 1);
		dim3 dimGridCubes( 1, 1, 1);
		/**Check collision with other clients*/
		cutilSafeCall(launch_CubeDinamic(BBdinamicCenter_d,(nrBBdinamic*2),BBdinamicSize_d,
										 xFuture,yFuture,zFuture,Id,okDinamic,dimGridCubes,dimBlock));
		cutilSafeCall(cudaThreadSynchronize());
		cutilSafeCall(cudaMemcpy(okDinamic, okDinamic_d, sizeof(int), cudaMemcpyDeviceToHost));
		checkCUDAError("invocare kernel");

	}
	else{
		hit = 1;
		printf("===============================================HIT=STATIC=============================================\n");
		printf("===============================================HIT=STATIC=============================================\n");
		//printf("===============================================HIT=STATIC=============================================\n");
		//printf("===============================================HIT=STATIC=============================================\n");
		//printf("===============================================HIT=STATIC=============================================\n");
		for(int i = 0 ; i < nrBBdinamic*6;i++)
			printf("%f  :",BBdinamic[i]);
	}
	/**If a collision has found modify the return value accordingly to that*/
	if (*okDinamic == 0)
	{
		//Update the client position in the array then return it to the client

	//	for (int i = 0; i < nrBBdinamic * 6; i += 1)
	//	{
	//		printf("BBdinamic %i : %f\n", i, BBdinamic[i]);
	//	}
	
		BBdinamicCenter[2 * Id] = xFuture;
		BBdinamicCenter[(2 * Id) + 1] = zFuture;
		float a, b, c, d;

		BBdinamic[6 * Id]; 
		a = BBdinamicCenter[2*Id] - size1/2;
		BBdinamic[6 * Id] = a;
		BBdinamic[(6 * Id) + 3]; 
		b = BBdinamicCenter[2*Id] + size1/2;
		BBdinamic[(6 * Id) + 3] = b; 
		c = BBdinamicCenter[(2*Id) + 1] - size2/2;
		BBdinamic[(6 * Id) + 2] = c; 
		d = BBdinamicCenter[(2*Id) + 1] + size2/2;
		BBdinamic[(6 * Id) + 5] = d ;

		//DO NOT DELETE THIS ONE!!! - SOME MORE TESTING TOMORROW MORNING IF I WAKE UP.
/*
		BBdinamic[6 * Id] = BBdinamicCenter[2*Id] - size1/2;
		BBdinamic[(6 * Id) + 3] = BBdinamicCenter[2*Id] + size1/2;
		BBdinamic[(6 * Id) + 2] = BBdinamicCenter[(2*Id) + 1] - size2/2;
		BBdinamic[(6 * Id) + 5] = BBdinamicCenter[(2*Id) + 1] + size2/2;
	*/	
	}
	else{
		//printf("===============================================HIT====================================================\n");
		//printf("===============================================HIT====================================================\n");
		//printf("===============================================HIT====================================================\n");
		//printf("===============================================HIT====================================================\n");
		//printf("===============================================HIT====================================================\n");
		hit = 1;
	}


	//for(int i = 0 ; i < nrBBdinamic *2; i++){
	//	printf("center: %f",BBdinamicCenter[i]);
	//}
    /**Do some cleaning*/
	cleanup();
	return hit;
}
/*!
 * The main function. 
 */
int main()
{		
	/////////////////////////////////////////////////////////////////////////////////
	BOOL bRet;
	HANDLE hRet;
	int counter = 0 ; 

	wsa_init();

	iocp = w_iocp_create();
	DIE(iocp == NULL, "w_iocp_create");

	/** Create server socket. */
	listenfd = tcp_create_listener(ECHO_LISTEN_PORT, DEFAULT_LISTEN_BACKLOG);
	DIE(listenfd == INVALID_SOCKET, "tcp_create_listener");

	hRet = w_iocp_add_handle(iocp, (HANDLE) listenfd);
	DIE(hRet != iocp, "w_iocp_add_handle");

	/** Use AcceptEx to schedule new connection acceptance. */
	create_iocp_accept();

	dlog(LOG_INFO, "Server waiting for connections on port %d\n", ECHO_LISTEN_PORT);

	/** Server main loop */
	while (1) {
		
		OVERLAPPED *ovp;
		ULONG_PTR key;
		DWORD bytes;
		//printf("Numarul: %d \n",counter++);
		/** Wait for overlapped I/O. */
		bRet = w_iocp_wait(iocp, &bytes, &key, &ovp);
		if (bRet == FALSE) {
			DWORD err;

			err = GetLastError();
			if (err == ERROR_NETNAME_DELETED) {
				connection_remove((struct connection *) key);
				continue;
			}
			DIE(bRet == FALSE, "w_iocp_wait");
		}

		/**
		 * Switch I/O notification types. Consider
		 *   - new connection requests (on server socket);
		 *   - socket communication (on connection sockets).
		 */

		if (key == listenfd) {
			dlog(LOG_DEBUG, "New connection\n");
			handle_new_connection(ovp);
		}
		else {
			
			handle_aio((struct connection *) key, bytes, ovp);
			
		}
	}

	wsa_cleanup();
	getch();
	system("PAUSE");
	return 0;

//////////////////////////////////////////////////////////////////////////////////
}
