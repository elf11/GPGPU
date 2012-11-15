#pragma once
#include "Matrix.h"

class MatrixStack : public Matrix
{
private:
	Matrix *stack;
	Matrix *current;
public:
	MatrixStack(int depth = 16);
	~MatrixStack();
	
	void Push();
	void Pop();
	void PopAll();
	
	inline void operator = (const Matrix &m)		{ *((Matrix *) this) = m; }
	inline void operator = (const MatrixStack &m)	{ *((Matrix *) this) = *((Matrix *) &m); }
};

