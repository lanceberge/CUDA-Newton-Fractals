newton: run.cu newton.cu complex.cu polynomial.cu
	make setup
	nvcc run.cu newton.cu complex.cu polynomial.cu -dc
	nvcc *.o -o bin/newton
	rm *.o

test: test.cu
	make setup
	nvcc test.cu newton.cu complex.cu polynomial.cu -dc
	nvcc *.o -o bin/test
	rm *.o
	./bin/test

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

runSmallTest:
	./bin/newton 100 100 smallTest
