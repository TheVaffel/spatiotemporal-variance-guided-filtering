
CFLAGS= -std=c++11 -Wall
LIBS = -lOpenCL -lOpenImageIO -lGL -fopenmp

svgf: svgf.cpp CLUtils/CLUtils.hpp CLUtils/CLUtils.cpp utils.cpp
	g++ -o $@ $^ $(LIBS) -I .  -Wno-ignored-attributes -g -m64 -DDEBUG -DUNIX
