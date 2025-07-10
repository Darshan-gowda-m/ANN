#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "backprop.h"

#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

void fastcopy(void *to, void *from, int len)
{
  char *_to = (char *)to;
  char *_from = (char *)from;
  for (int _i = 0; _i < len; _i++)
    *_to++ = *_from++;
}

double drnd()
{
  return ((double)rand() / (double)BIGRND);
}

double dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

double squash(double x)
{
  return (1.0 / (1.0 + exp(-x)));
}

double *alloc_1d_dbl(int n)
{
  double *new = (double *)malloc(n * sizeof(double));
  if (!new)
  {
    fprintf(stderr, "ALLOC_1D_DBL: Couldn't allocate array\n");
    return NULL;
  }
  return new;
}

double **alloc_2d_dbl(int m, int n)
{
  double **new = (double **)malloc(m * sizeof(double *));
  if (!new)
  {
    fprintf(stderr, "ALLOC_2D_DBL: Couldn't allocate array\n");
    return NULL;
  }

  for (int i = 0; i < m; i++)
  {
    new[i] = alloc_1d_dbl(n);
    if (!new[i])
      return NULL;
  }
  return new;
}

void bpnn_randomize_weights(double **w, int m, int n)
{
  for (int i = 0; i <= m; i++)
    for (int j = 0; j <= n; j++)
      w[i][j] = dpn1();
}

void bpnn_zero_weights(double **w, int m, int n)
{
  for (int i = 0; i <= m; i++)
    for (int j = 0; j <= n; j++)
      w[i][j] = 0.0;
}

void bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet = (BPNN *)malloc(sizeof(BPNN));
  if (!newnet)
  {
    fprintf(stderr, "BPNN_CREATE: Couldn't allocate network\n");
    return NULL;
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;

  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);
  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return newnet;
}

void bpnn_free(BPNN *net)
{
  if (!net)
    return;

  free(net->input_units);
  free(net->hidden_units);
  free(net->output_units);
  free(net->hidden_delta);
  free(net->output_delta);
  free(net->target);

  for (int i = 0; i <= net->input_n; i++)
  {
    free(net->input_weights[i]);
    free(net->input_prev_weights[i]);
  }
  free(net->input_weights);
  free(net->input_prev_weights);

  for (int i = 0; i <= net->hidden_n; i++)
  {
    free(net->hidden_weights[i]);
    free(net->hidden_prev_weights[i]);
  }
  free(net->hidden_weights);
  free(net->hidden_prev_weights);

  free(net);
}

BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet = bpnn_internal_create(n_in, n_hidden, n_out);
  if (!newnet)
    return NULL;

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);

  return newnet;
}

void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
  l1[0] = 1.0;
  for (int j = 1; j <= n2; j++)
  {
    double sum = 0.0;
    for (int k = 0; k <= n1; k++)
      sum += conn[k][j] * l1[k];
    l2[j] = squash(sum);
  }
}

void bpnn_output_error(double *delta, double *target, double *output, int nj, double *err)
{
  double errsum = 0.0;
  for (int j = 1; j <= nj; j++)
  {
    double o = output[j], t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no,
                       double **who, double *hidden, double *err)
{
  double errsum = 0.0;
  for (int j = 1; j <= nh; j++)
  {
    double h = hidden[j], sum = 0.0;
    for (int k = 1; k <= no; k++)
      sum += delta_o[k] * who[j][k];
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly,
                         double **w, double **oldw, double eta, double momentum)
{
  ly[0] = 1.0;
  for (int j = 1; j <= ndelta; j++)
  {
    for (int k = 0; k <= nly; k++)
    {
      double new_dw = (eta * delta[j] * ly[k]) + (momentum * oldw[k][j]);
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}

void bpnn_feedforward(BPNN *net)
{
  bpnn_layerforward(net->input_units, net->hidden_units,
                    net->input_weights, net->input_n, net->hidden_n);
  bpnn_layerforward(net->hidden_units, net->output_units,
                    net->hidden_weights, net->hidden_n, net->output_n);
}

void bpnn_train(BPNN *net, double eta, double momentum, double *eo, double *eh)
{
  bpnn_feedforward(net);

  bpnn_output_error(net->output_delta, net->target, net->output_units,
                    net->output_n, eo);
  bpnn_hidden_error(net->hidden_delta, net->hidden_n, net->output_delta,
                    net->output_n, net->hidden_weights, net->hidden_units, eh);

  bpnn_adjust_weights(net->output_delta, net->output_n, net->hidden_units,
                      net->hidden_n, net->hidden_weights,
                      net->hidden_prev_weights, eta, momentum);
  bpnn_adjust_weights(net->hidden_delta, net->hidden_n, net->input_units,
                      net->input_n, net->input_weights,
                      net->input_prev_weights, eta, momentum);
}

void bpnn_save(BPNN *net, char *filename)
{
  FILE *fd = fopen(filename, "wb");
  if (!fd)
  {
    perror("BPNN_SAVE");
    return;
  }

  printf("Saving %dx%dx%d network to '%s'\n",
         net->input_n, net->hidden_n, net->output_n, filename);

  fwrite(&net->input_n, sizeof(int), 1, fd);
  fwrite(&net->hidden_n, sizeof(int), 1, fd);
  fwrite(&net->output_n, sizeof(int), 1, fd);

  for (int i = 0; i <= net->input_n; i++)
    fwrite(net->input_weights[i], sizeof(double), net->hidden_n + 1, fd);

  for (int i = 0; i <= net->hidden_n; i++)
    fwrite(net->hidden_weights[i], sizeof(double), net->output_n + 1, fd);

  fclose(fd);
}

BPNN *bpnn_read(char *filename)
{
  FILE *fd = fopen(filename, "rb");
  if (!fd)
    return NULL;

  int n1, n2, n3;
  fread(&n1, sizeof(int), 1, fd);
  fread(&n2, sizeof(int), 1, fd);
  fread(&n3, sizeof(int), 1, fd);

  BPNN *new = bpnn_internal_create(n1, n2, n3);
  if (!new)
  {
    fclose(fd);
    return NULL;
  }

  for (int i = 0; i <= n1; i++)
    fread(new->input_weights[i], sizeof(double), n2 + 1, fd);

  for (int i = 0; i <= n2; i++)
    fread(new->hidden_weights[i], sizeof(double), n3 + 1, fd);

  fclose(fd);
  return new;
}