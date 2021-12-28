newton:
	make setup
	nvcc src/main.cu src/newton.cu src/complex.cu src/polynomial.cu src/png_util.c -O3 -lpng -o bin/newton -rdc=true

debug:
	nvcc src/main.cu src/newton.cu src/complex.cu src/polynomial.cu src/png_util.c -lpng -g -G -o bin/newton -rdc=true

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

movie:
	./bin/newton ${name} step=true
	make stitchMovie

stitchMovie:
	ffmpeg -y -start_number 0 -r 24 -i fractals/${name}Step-%d.png -b:v 8192k -c:v mpeg4 fractals/${name}.mp4
	rm fractals/*Step*
