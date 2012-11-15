#ifndef IMAGE_H
#define IMAGE_H

struct Image {
    unsigned long sizeX;
    unsigned long sizeY;
    char *data;
};

int ImageLoad(char *filename, Image *image);

#endif