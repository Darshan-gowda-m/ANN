#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pgmimage.h"
#include "backprop.h"

int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    fprintf(stderr, "Usage: %s net-file image-file x y hidden-unit-num\n", argv[0]);
    return 1;
  }

  BPNN *net = bpnn_read(argv[1]);
  if (!net)
  {
    fprintf(stderr, "Can't read network file '%s'\n", argv[1]);
    return 1;
  }

  int nc = atoi(argv[3]), nr = atoi(argv[4]), h = atoi(argv[5]);
  IMAGE *img = img_creat(argv[2], nr, nc);
  if (!img)
  {
    fprintf(stderr, "Can't create image file '%s'\n", argv[2]);
    bpnn_free(net);
    return 1;
  }

  double maxwt = -1e6, minwt = 1e6;
  for (int i = 0; i < nr * nc; i++)
  {
    double wt = net->input_weights[i][h];
    if (wt > maxwt)
      maxwt = wt;
    if (wt < minwt)
      minwt = wt;
  }
  double range = maxwt - minwt;

  for (int i = 0; i < nr * nc; i++)
  {
    int pxl = (int)(((net->input_weights[i][h] - minwt) / range) * 255.0);
    img_setpixel(img, i / nc, i % nc, pxl);
  }

  img_write(img, argv[2]);
  img_free(img);
  bpnn_free(net);
  return 0;
}