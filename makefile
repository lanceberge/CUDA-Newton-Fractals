CPPFLAGS=-O3 -lpng -rdc=true
CPPFILES=src/main.cu src/newton.cu src/complex.cu src/polynomial.cu src/png_util.c
OUT=bin/newton

newton:
	make setup
	nvcc ${CPPFILES} ${CPPFLAGS} -o ${OUT}


debug:
	nvcc ${CPPFILES} ${CPPFLAGS} -g -G -o ${OUT}

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

runAll:
	make runOrder7
	make runOrder12

runOrder7:
	./bin/newton order7

runOrder12:
	./bin/newton order12

name = order7
args = ""

movie:
	./bin/newton ${name} ${args} step=true
	make stitchMovie

stitchMovie:
	ffmpeg -y -start_number 0 -r 24 -i fractals/${name}Step-%d.png -b:v 8192k -c:v mpeg4 fractals/${name}.mp4
	rm fractals/*Step*
