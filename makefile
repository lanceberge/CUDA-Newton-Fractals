## TODO: step functionality

newton:
	make setup
	nvcc src/main.cu src/newton.cu src/complex.cu src/polynomial.cu -O3 -lpng -o bin/newton -rdc=true

test:
	make setup
	nvcc src/test.cu src/newton.cu src/complex.cu src/polynomial.cu -o bin/test -rdc=true
	./bin/test

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

runAll:
	make runSmallTest
	make runBigTest
	make runBigTest2

runSmallTest:
	./bin/newton 100 100 smallTest

runBigTest:
	./bin/newton 500 500 bigTest


runBigTest2:
	./bin/newton 500 500 bigTest2

name = bigTest

movie:
	ffmpeg -y -start_number 0 -r 24 -i plots/$(name)Step-%d.png -b:v 8192k -c:v mpeg4 plots/$(name).mp4
	rm plots/*Step*
