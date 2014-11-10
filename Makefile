OPENCVFLAGS=`pkg-config --cflags --libs opencv`

all: rl_deconv

rl_deconv: rl_deconv.cpp
	g++ -g rl_deconv.cpp -o rl_deconv $(OPENCVFLAGS)

rl_deconv_c: rl_deconv.c
	gcc -g rl_deconv.c -o rl_deconv_c $(OPENCVFLAGS)
