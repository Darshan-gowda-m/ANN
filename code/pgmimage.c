#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgmimage.h"

char *img_basename(char *filename)
{
  char *new, *part;
  int len = strlen(filename), dex = len - 1;

  while (dex > -1 && filename[dex] != '/')
    dex--;
  dex++;

  part = &filename[dex];
  len = strlen(part);
  new = (char *)malloc(len + 1);
  if (new)
    strcpy(new, part);
  return new;
}

IMAGE *img_alloc()
{
  IMAGE *new = (IMAGE *)malloc(sizeof(IMAGE));
  if (new)
  {
    new->rows = new->cols = 0;
    new->name = NULL;
    new->data = NULL;
  }
  return new;
}

IMAGE *img_creat(char *name, int nr, int nc)
{
  IMAGE *new = img_alloc();
  if (!new)
    return NULL;

  new->data = (int *)malloc(nr * nc * sizeof(int));
  if (!new->data)
  {
    free(new);
    return NULL;
  }

  new->name = img_basename(name);
  new->rows = nr;
  new->cols = nc;

  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nc; j++)
      img_setpixel(new, i, j, 0);

  return new;
}

void img_free(IMAGE *img)
{
  if (img)
  {
    free(img->data);
    free(img->name);
    free(img);
  }
}

void img_setpixel(IMAGE *img, int r, int c, int val)
{
  if (img && img->data)
    img->data[(r * img->cols) + c] = val;
}

int img_getpixel(IMAGE *img, int r, int c)
{
  return (img && img->data) ? img->data[(r * img->cols) + c] : 0;
}

IMAGE *img_open(char *filename)
{
  FILE *pgm = fopen(filename, "r");
  if (!pgm)
  {
    perror("IMGOPEN");
    return NULL;
  }

  IMAGE *new = img_alloc();
  if (!new)
  {
    fclose(pgm);
    return NULL;
  }

  char line[512];
  int type, nc, nr, maxval;

  fgets(line, sizeof(line), pgm);
  sscanf(line, "P%d", &type);
  if (type != 5 && type != 2)
  {
    fprintf(stderr, "IMGOPEN: Only handles pgm files (type P5 or P2)\n");
    fclose(pgm);
    img_free(new);
    return NULL;
  }

  fgets(line, sizeof(line), pgm);
  sscanf(line, "%d %d", &nc, &nr);
  new->rows = nr;
  new->cols = nc;

  fgets(line, sizeof(line), pgm);
  sscanf(line, "%d", &maxval);
  if (maxval > 255)
  {
    fprintf(stderr, "IMGOPEN: Only handles 8-bit pgm files\n");
    fclose(pgm);
    img_free(new);
    return NULL;
  }

  new->data = (int *)malloc(nr * nc * sizeof(int));
  if (!new->data)
  {
    fclose(pgm);
    img_free(new);
    return NULL;
  }

  if (type == 5)
  {
    for (int i = 0; i < nr; i++)
      for (int j = 0; j < nc; j++)
        img_setpixel(new, i, j, fgetc(pgm));
  }
  else
  {
    for (int i = 0; i < nr; i++)
    {
      for (int j = 0; j < nc; j++)
      {
        int val;
        fscanf(pgm, "%d", &val);
        img_setpixel(new, i, j, val);
      }
    }
  }

  fclose(pgm);
  new->name = img_basename(filename);
  return new;
}

int img_write(IMAGE *img, char *filename)
{
  FILE *iop = fopen(filename, "w");
  if (!iop)
  {
    perror("IMG_WRITE");
    return 0;
  }

  fprintf(iop, "P2\n%d %d\n255\n", img->cols, img->rows);

  int k = 1;
  for (int i = 0; i < img->rows; i++)
  {
    for (int j = 0; j < img->cols; j++)
    {
      int val = img_getpixel(img, i, j);
      if (val < 0 || val > 255)
      {
        fprintf(stderr, "IMG_WRITE: Clamping value %d at (%d,%d)\n", val, i, j);
        val = (val < 0) ? 0 : 255;
      }
      fprintf(iop, "%d%c", val, (k++ % 10) ? ' ' : '\n');
    }
  }

  fclose(iop);
  return 1;
}

IMAGELIST *imgl_alloc()
{
  IMAGELIST *new = (IMAGELIST *)malloc(sizeof(IMAGELIST));
  if (new)
  {
    new->n = 0;
    new->list = NULL;
  }
  return new;
}

void imgl_add(IMAGELIST *il, IMAGE *img)
{
  if (!il || !img)
    return;

  il->list = (IMAGE **)realloc(il->list, (il->n + 1) * sizeof(IMAGE *));
  if (il->list)
  {
    il->list[il->n++] = img;
  }
}

void imgl_free(IMAGELIST *il)
{
  if (il)
  {
    free(il->list);
    free(il);
  }
}

void imgl_munge_name(char *buf)
{
  int j = 0;
  while (buf[j] && buf[j] != '\n')
    j++;
  buf[j] = '\0';
}

void imgl_load_images_from_textfile(IMAGELIST *il, char *filename)
{
  FILE *fp = fopen(filename, "r");
  if (!fp)
  {
    perror("IMGL_LOAD_IMAGES_FROM_TEXTFILE");
    return;
  }

  char buf[2000];
  while (fgets(buf, sizeof(buf), fp))
  {
    imgl_munge_name(buf);
    printf("Loading '%s'...", buf);
    fflush(stdout);

    IMAGE *iimg = img_open(buf);
    if (iimg)
    {
      imgl_add(il, iimg);
      printf("done\n");
    }
    else
    {
      printf("failed\n");
    }
  }

  fclose(fp);
}