newton: run.cu newton.cu complex.cu polynomial.cu
	make setup
	nvcc run.cu newton.cu complex.cu polynomial.cu -dc
	nvcc *.o -o bin/newton
	rm *.o

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

runSmallTest:
	./bin/newton 100 100 smallTest
