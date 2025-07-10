#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgmimage.h"
#include "backprop.h"

#define TARGET_HIGH 0.9
#define TARGET_LOW 0.1

/**
 * Sets the target output based on presence of "_sunglasses_" in image filename.
 */
void load_target(IMAGE *img, BPNN *net)
{
  const char *name = NAME(img);
  if (strstr(name, "_sunglasses_") != NULL)
  {
    net->target[1] = TARGET_HIGH; // Sunglasses present
  }
  else
  {
    net->target[1] = TARGET_LOW; // No sunglasses
  }
}

/**
 * Loads image pixel values into the input layer of the network.
 */
void load_input_with_image(IMAGE *img, BPNN *net)
{
  int rows = ROWS(img);
  int cols = COLS(img);
  int imgsize = rows * cols;

  if (imgsize != net->input_n)
  {
    fprintf(stderr, "LOAD_INPUT_WITH_IMAGE: Image has %d pixels, but network expects %d input units.\n",
            imgsize, net->input_n);
    exit(EXIT_FAILURE);
  }

  double *units = net->input_units;
  int k = 1;

  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      units[k++] = img_getpixel(img, i, j) / 255.0;
    }
  }
}
