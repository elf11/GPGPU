#include "MatrixStack.h"

MatrixStack::MatrixStack(int depth)
{
	stack	= new Matrix[depth];
	current = stack;
	*(this) = Matrix::Identity;
}

MatrixStack::~MatrixStack()
{
	delete [] stack;
}

void MatrixStack::Push()
{
	*current = *this;
	++current;
}

void MatrixStack::Pop()
{
	if(current > stack)
	{
		--current;
		*this = *current;
	}
}

void MatrixStack::PopAll()
{
	if(current > stack)
	{
		current = stack;
		*this = *current;
	}
}
