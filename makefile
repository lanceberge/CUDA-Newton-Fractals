CPPFLAGS=-O3 -lpng -rdc=true
CPPFILES=src/main.cu src/newton.cu src/complex.cu src/polynomial.cu src/png_util.c
OUT     =bin/newton

newton: ${CPPFILES}
	make setup
	nvcc ${CPPFILES} ${CPPFLAGS} -o ${OUT}

debug: ${CPPFILES}
	nvcc ${CPPFILES} ${CPPFLAGS} -g -G -o ${OUT}

test: ${CPPFILES}
	nvcc ${CPPFILES} ${CPPFLAGS} -g -G -DhostSolns -o ${OUT}

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

runOrder7: ${OUT}
	./bin/newton order7

runOrder12: ${OUT}
	./bin/newton order12

runAll: ${OUT}
	make runOrder7
	make runOrder12

name = order7
args = ""

movie: ${OUT}
	./bin/newton ${name} ${args} step=true
	make stitchMovie

stitchMovie:
	ffmpeg -y -start_number 0 -r 24 -i fractals/${name}Step-%d.png -b:v 8192k -c:v mpeg4 fractals/${name}.mp4
	rm fractals/*Step*
