CFLAGS = -g -I. -Wall -Wextra -std=gnu99
CFLAGS = -g -Wall -Wextra -std=gnu99

LFLAGS = -lm

all: facetrain hidtopgm outtopgm

facetrain: facetrain.o imagenet.o backprop_initr.o pgmimage.o
	$(CC) $^ $(LFLAGS) -o $@

hidtopgm: hidtopgm.o pgmimage.o backprop_initr.o
	$(CC) $^ $(LFLAGS) -o $@

outtopgm: outtopgm.o pgmimage.o backprop_initr.o
	$(CC) $^ $(LFLAGS) -o $@

backprop_initr.o: backprop.c backprop.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o facetrain hidtopgm outtopgm
