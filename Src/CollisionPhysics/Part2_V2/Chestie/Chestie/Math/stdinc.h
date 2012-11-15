#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "RMath.h"
//#include "platform.h"
//#include "debug.h"
//#include "misc.h"

#define M_PIf ((float)M_PI)


typedef char s8;
typedef unsigned char u8;

typedef char sbyte;
typedef unsigned char byte;

typedef short s16;
typedef unsigned short u16;

typedef int s32;
typedef unsigned int u32;

typedef float f32;

#if WIN32
#define ASSERTD(x, y)	winassert((y), (x))
//#define ASSERTD(x, y)	{if (!(x)){int i = 0; int a = 1/i; i+= a;}}
#else
#define ASSERTD(x, y)   {if (!(x)){int i = 0; int a = 1/i; i+= a;}}
#endif

#ifndef ASSERT
#define ASSERT(cond) ASSERTD(cond, "")
#endif

#define ASSERT_RANGE(v1, x, v2) ASSERT( (v1 <= x) && (x < v2) )
#define ASSERT_ERR(msg) ASSERTD(0, msg)
#define TRACE_CALL() DEBUG_OUT("TRACE_CALL: %s() in %s:%d\n", __FUNCTION__, __FILE__, __LINE__)


#define SDK_DECLAREARRAY_2D(TYPE,NAME)	 TYPE **NAME;
#define SDK_CREATEARRAY_2D(TYPE,NAME,XSIZE,YSIZE)	NAME = new TYPE*[(XSIZE)]; for (int tempi = 0; tempi < (XSIZE); tempi++) NAME[tempi] = new TYPE[(YSIZE)];
#define SDK_DESTROYARRAY_2D(NAME,XSIZE)	 for (int tempi = 0; tempi < (XSIZE); tempi++) delete[] NAME[tempi]; delete[] NAME;

#define SDK_DECLAREARRAY_1D(TYPE,NAME)	 TYPE *NAME;
#define SDK_CREATEARRAY_1D(TYPE,NAME,XSIZE)	 NAME = new TYPE[(XSIZE)];
#define SDK_DESTROYARRAY_1D(NAME)	 delete[] NAME;


#define SAFE_DEL(x) {if (x) {delete x; x = NULL;}}
#define SAFE_DEL_ARRAY(x) {if (x) {delete[] x; x = NULL;}}


#define MOFFSET(type, member) ((size_t)&reinterpret_cast<const volatile char&>(((type*)0)->member))

#define ALLOC_STACK(size) alloca(size)

#define STRDUPss(x) strdup2(x)

#define EnableFlag(var, flag)		(var) = (var) | (flag)
#define DisableFlag(var,flag)		(var) = (var) & (~(flag))
#define IsFlagEnabled(var,flag)		(((var) & (flag)) ? 1: 0)


