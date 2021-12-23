newton:
	make setup
	nvcc src/main.cu src/newton.cu src/complex.cu src/polynomial.cu -O3 -o bin/newton -rdc=true

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
	make runBigTest3

runSmallTest:
	./bin/newton 100 100 smallTest step
	./bin/newton 100 100 smallTest

runBigTest:
	./bin/newton 200 200 bigTest step
	./bin/newton 500 500 bigTest


runBigTest2:
	./bin/newton 200 200 bigTest2 step
	./bin/newton 500 500 bigTest2

runBigTest3:
	./bin/newton 200 200 bigTest3 step
	./bin/newton 500 500 bigTest3
	./bin/newton 200 200 bigTest3L1 step L1
	./bin/newton 500 500 bigTest3L1 L1
