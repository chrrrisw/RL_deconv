OPENCVFLAGS=`pkg-config --cflags --libs opencv`

all: rl_deconv

rl_deconv: rl_deconv.cpp
	g++ -g rl_deconv.cpp -o rl_deconv $(OPENCVFLAGS)