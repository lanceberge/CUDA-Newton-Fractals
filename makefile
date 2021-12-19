newton:
	make setup
	nvcc src/main.cu src/newton.cu src/complex.cu src/polynomial.cu -o bin/newton -rdc=true

test:
	make setup
	nvcc src/test.cu src/newton.cu src/complex.cu src/polynomial.cu -o bin/test -rdc=true
	./bin/test

setup:
	if [ ! -d "./bin" ]; then \
	mkdir bin; \
	fi

debug:
	nvcc -g -G src/test.cu src/newton.cu src/complex.cu src/polynomial.cu -dc
	nvcc -g -G *.o -o bin/test
	rm *.o

runAll:
	make runSmallTest
	make runBigTest
	make runBigTest2
	make runBigTest3

runSmallTest:
	./bin/newton 100 100 smallTest
	./bin/newton 100 100 smallTest step

runBigTest:
	./bin/newton 500 500 bigTest
	./bin/newton 200 200 bigTest step


runBigTest2:
	./bin/newton 500 500 bigTest2
	./bin/newton 200 200 bigTest2 step

runBigTest3:
	./bin/newton 500 500 bigTest3
	./bin/newton 200 200 bigTest3 step
	./bin/newton 500 500 bigTest3L1 L1
	./bin/newton 200 200 bigTest3L1 step L1
