/*! \file thread.cpp 
 *\brief Work in Progress
 */
#include "thread.h"


/**Creates a thread
* Function should be called from the main thread
* Parametres:
*	- hthread : the new thread
*	- IDThread : will be a return value,contains the thread id(a random number ,not to be used as an index) 
*/
void CreateThread(HANDLE hThread,DWORD IDThread)
{
	hThread = CreateThread(NULL,                /* default security attributes */
			      0,                                   /* default stack size */
			      (LPTHREAD_START_ROUTINE) ThreadFunc, /* routine to execute */
			      NULL,                                /* no thread parameter */
			      0,                                   /* immediately run the thread */
			      &IDThread);                          /* thread id */
}



/** This function represents the word that can be doen by a thread
* Function is called when the thread is created
*Parametres:
*	- lpPatametres : based on the funciotn syntax,not important
*/
DWORD WINAPI ThreadFunc(LPVOID lpParameter)
{
	/*TODO: write the work that should be done by a thread. 
	* I was thinking of smth like:
	*	- get current command : see output from the socket read action
	*	- process current command : check for collisions and eventually update position 
	*	- send back the result : send a message containing the result you've got
	*PLEASE : mail me your suggestions
	*/
	return 0;
}




/**Waiting for threads to complete their work
* Function should be called from the main thread
* Parametres :
*	- hThread[] - vector of current threads
*	- threadNo	- number of threads 
*/
void WaitThreads(HANDLE hThread[],int threadNo)
{	DWORD dwReturn;
	for (int i = 0; i < threadNo; i++) {
		dwReturn = WaitForSingleObject(hThread[i], INFINITE);
		//DIE(dwReturn == WAIT_FAILED, "WaitForSingleObject");
	}
}


/* * Cleaning up after we don't need the threads any longer
* Function should be called from the main thread
* Parametres :
*	- hThread[] - vector of current threads
*	- threadNo	- number of threads
*/

void cleanUp(HANDLE hThread[],int threadNo)
{	
	DWORD dwRet;
	for (int i = 0; i < threadNo; i++){
		dwRet = CloseHandle(hThread[i]);
		//DIE(dwRet == FALSE, "CloseHandle");
	}
}
