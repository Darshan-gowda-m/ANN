// imagenet.h
#ifndef IMAGENET_H
#define IMAGENET_H

#include "pgmimage.h"
#include "backprop.h"

// Declare functions used across files
void load_input_with_image(IMAGE *img, BPNN *net);
void load_target(IMAGE *img, BPNN *net);

#endif
