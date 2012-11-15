#pragma once

class RectangleInt
{
public:
	int x;
	int y;
	int width;
	int height;
	
    inline int      Area()          const { return width * height; }
    inline float    Aspect()        const { return float(width) / float(height); }
    inline float    OneOverAspect() const { return float(height) / float(width); }
    inline bool     IsLandscape()   const { return width >= height; }
    inline bool     IsPortrait()    const { return height > width;  }
    
    inline void     ScaleSizeWithAspect(const int refSize, int &sizeWidth, int &sizeHeight) const
    {
        if(width >= height)
        {
            sizeWidth  = refSize;
            sizeHeight = (refSize * height) / width;
        }
        else
        {
            sizeWidth  = (refSize * width) / height;
            sizeHeight = refSize;
        }
        if (sizeWidth % 2)  ++sizeWidth;
        if (sizeHeight % 2) ++sizeHeight;
    }
    
	inline RectangleInt() {}
	inline RectangleInt(const int X, const int Y, const int Width, const int Height) : x(X), y(Y), width(Width), height(Height) {}

	inline bool IsPointInside(int pX,int pY)
	{
		if (pX>=x && pY >=y && pX <=x+width && pY<=y+height) return true;
		
		return false;
	}
	inline void ClampPoint(float &pX,float &pY)
	{
		if (pX >= x + width) pX = x + width;
		if (pX <= x) pX = x;
		if (pY >= y + height) pY = y + height;
		if (pY <= y) pY = y;
	}
};
