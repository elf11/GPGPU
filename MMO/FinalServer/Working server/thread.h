#undef _UNICODE
#undef UNICODE

#include<windows.h>
#include<stdio.h>

//#include "API_Functions.h"

void CreateThread(HANDLE hThread,DWORD IDThread);
DWORD WINAPI ThreadFunc(LPVOID lpParameter);
void WaitThreads(int threadNo,HANDLE hThread[]);
void cleanUp(HANDLE hThread[]);



