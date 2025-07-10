#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pgmimage.h"
#include "backprop.h"

int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    fprintf(stderr, "Usage: %s net-file image-file x y output-unit-num\n", argv[0]);
    return 1;
  }

  // Read command line arguments
  char *net_file = argv[1];
  char *image_file = argv[2];
  int nc = atoi(argv[3]);  // Number of columns
  int nr = atoi(argv[4]);  // Number of rows
  int out = atoi(argv[5]); // Output unit index

  // Load neural network
  BPNN *net = bpnn_read(net_file);
  if (!net)
  {
    fprintf(stderr, "Can't read network file '%s'\n", net_file);
    return 1;
  }

  // Create output image of double resolution
  IMAGE *img = img_creat(image_file, 2 * nr, 2 * nc);
  if (!img)
  {
    fprintf(stderr, "Can't create image file '%s'\n", image_file);
    bpnn_free(net);
    return 1;
  }

  // Print hidden-to-output weights for the specified output unit
  printf("Output unit %d weights:\n", out);
  for (int i = 0; i < net->hidden_n; i++)
  {
    printf("%g\n", net->hidden_weights[i][out]);
  }

  // Find min and max weights for normalization
  double maxwt = -1e6, minwt = 1e6;
  for (int i = 0; i < nr * nc; i++)
  {
    double wt = net->input_weights[i][out];
    if (wt > maxwt)
      maxwt = wt;
    if (wt < minwt)
      minwt = wt;
  }

  double range = maxwt - minwt;
  if (range == 0.0)
    range = 1.0; // Prevent division by zero

  // Create grayscale image of weights (upsampled by 2x)
  for (int i = 0; i < nr; i++)
  {
    for (int j = 0; j < nc; j++)
    {
      int idx = i * nc + j;
      int pxl = (int)(((net->input_weights[idx][out] - minwt) / range) * 255.0);
      img_setpixel(img, 2 * i, 2 * j, pxl);
      img_setpixel(img, 2 * i + 1, 2 * j + 1, pxl);
      img_setpixel(img, 2 * i, 2 * j + 1, pxl);
      img_setpixel(img, 2 * i + 1, 2 * j, pxl);
    }
  }

  // Save the PGM image
  img_write(img, image_file);

  // Clean up
  img_free(img);
  bpnn_free(net);
  return 0;
}
