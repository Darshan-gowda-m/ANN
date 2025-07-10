#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pgmimage.h"
#include "backprop.h"
#include "imagenet.h"

void printusage(char *prog)
{
  printf("USAGE: %s\n", prog);
  printf("       -n <network file>\n");
  printf("       [-e <number of epochs>]\n");
  printf("       [-s <random number generator seed>]\n");
  printf("       [-S <number of epochs between saves>]\n");
  printf("       [-t <training set list>]\n");
  printf("       [-1 <testing set 1 list>]\n");
  printf("       [-2 <testing set 2 list>]\n");
  printf("       [-T (test only mode)]\n");
}

int evaluate_performance(BPNN *net, double *err)
{
  double delta = net->target[1] - net->output_units[1];
  *err = 0.5 * delta * delta;
  return (net->target[1] > 0.5) ? (net->output_units[1] > 0.5) : (net->output_units[1] <= 0.5);
}

void performance_on_imagelist(BPNN *net, IMAGELIST *il, int list_errors)
{
  double err = 0.0;
  int correct = 0;

  for (int i = 0; i < il->n; i++)
  {
    load_input_with_image(il->list[i], net);
    bpnn_feedforward(net);
    load_target(il->list[i], net);

    double val;
    if (evaluate_performance(net, &val))
    {
      correct++;
    }
    else if (list_errors)
    {
      printf("%s - outputs ", NAME(il->list[i]));
      for (int j = 1; j <= net->output_n; j++)
        printf("%.3f ", net->output_units[j]);
      putchar('\n');
    }
    err += val;
  }

  if (!list_errors)
  {
    printf("%g %g ", (il->n > 0) ? ((double)correct / il->n) * 100.0 : 0.0,
           (il->n > 0) ? err / il->n : 0.0);
  }
}

void backprop_face(IMAGELIST *trainlist, IMAGELIST *test1list,
                   IMAGELIST *test2list, int epochs, int savedelta,
                   char *netname, int list_errors)
{
  BPNN *net = bpnn_read(netname);
  if (!net && trainlist->n > 0)
  {
    printf("Creating new network '%s'\n", netname);
    IMAGE *iimg = trainlist->list[0];
    int imgsize = ROWS(iimg) * COLS(iimg);
    net = bpnn_create(imgsize, 4, 1);
  }

  if (!net)
  {
    fprintf(stderr, "Need some images to train on, use -t\n");
    return;
  }

  if (epochs > 0)
  {
    printf("Training for %d epochs\n", epochs);
    printf("Saving every %d epochs\n", savedelta);
  }

  printf("0 0.0 ");
  performance_on_imagelist(net, trainlist, 0);
  performance_on_imagelist(net, test1list, 0);
  performance_on_imagelist(net, test2list, 0);
  printf("\n");

  if (list_errors)
  {
    printf("\nTraining errors:\n");
    performance_on_imagelist(net, trainlist, 1);
    printf("\nTest1 errors:\n");
    performance_on_imagelist(net, test1list, 1);
    printf("\nTest2 errors:\n");
    performance_on_imagelist(net, test2list, 1);
  }

  for (int epoch = 1; epoch <= epochs; epoch++)
  {
    printf("%d ", epoch);
    double sumerr = 0.0;

    for (int i = 0; i < trainlist->n; i++)
    {
      load_input_with_image(trainlist->list[i], net);
      load_target(trainlist->list[i], net);
      double out_err, hid_err;
      bpnn_train(net, 0.3, 0.3, &out_err, &hid_err);
      sumerr += out_err + hid_err;
    }

    printf("%g ", sumerr);
    performance_on_imagelist(net, trainlist, 0);
    performance_on_imagelist(net, test1list, 0);
    performance_on_imagelist(net, test2list, 0);
    printf("\n");

    if (epoch % savedelta == 0)
      bpnn_save(net, netname);
  }

  if (epochs > 0)
    bpnn_save(net, netname);
}

int main(int argc, char *argv[])
{
  char netname[256] = {0};
  char trainname[256] = {0};
  char test1name[256] = {0};
  char test2name[256] = {0};
  int epochs = 100, seed = 102194, savedelta = 100, list_errors = 0;

  if (argc < 2)
  {
    printusage(argv[0]);
    return 1;
  }

  IMAGELIST *trainlist = imgl_alloc();
  IMAGELIST *test1list = imgl_alloc();
  IMAGELIST *test2list = imgl_alloc();

  for (int i = 1; i < argc; i++)
  {
    if (argv[i][0] == '-')
    {
      switch (argv[i][1])
      {
      case 'n':
        strncpy(netname, argv[++i], sizeof(netname) - 1);
        break;
      case 'e':
        epochs = atoi(argv[++i]);
        break;
      case 's':
        seed = atoi(argv[++i]);
        break;
      case 'S':
        savedelta = atoi(argv[++i]);
        break;
      case 't':
        strncpy(trainname, argv[++i], sizeof(trainname) - 1);
        break;
      case '1':
        strncpy(test1name, argv[++i], sizeof(test1name) - 1);
        break;
      case '2':
        strncpy(test2name, argv[++i], sizeof(test2name) - 1);
        break;
      case 'T':
        list_errors = 1;
        epochs = 0;
        break;
      default:
        fprintf(stderr, "Unknown switch '%c'\n", argv[i][1]);
        break;
      }
    }
  }

  if (trainname[0])
    imgl_load_images_from_textfile(trainlist, trainname);
  if (test1name[0])
    imgl_load_images_from_textfile(test1list, test1name);
  if (test2name[0])
    imgl_load_images_from_textfile(test2list, test2name);

  if (!netname[0])
  {
    fprintf(stderr, "Must specify output file with -n\n");
    return 1;
  }

  bpnn_initialize(seed);
  printf("%d training, %d test1, %d test2 images\n",
         trainlist->n, test1list->n, test2list->n);

  backprop_face(trainlist, test1list, test2list, epochs, savedelta,
                netname, list_errors);

  return 0;
}