all:
	mpic++ -std=c++11 -O2 -Wall -Werror -fopenmp -c q_gates.cpp
	ar r libq_gates.a q_gates.o

run_test: test
	mpirun -np 1 ./test
	mpirun -np 2 ./test
	mpirun -np 4 ./test

test: test.o
	mpic++ -std=c++11 -fopenmp -O2 -Wall -Werror -o test test.o -L. -lq_gates

test.o: test.cpp
	mpic++ -std=c++11 -O2 -Wall -Werror -fopenmp -c test.cpp

clean:
	rm *.o test lib*

cpplint: q_gates.cpp test.cpp
	cpplint --linelength=100 --filter=-legal,-build,-read q_gates.cpp test.cpp