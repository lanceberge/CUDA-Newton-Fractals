newton: newton.cu
	make setup
	nvcc -o bin/newton run.cu newton.cu polynomial.cu --expt-relaxed-constexpr

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

runSmallTest:
	./bin/newton 100 100 smallTest
