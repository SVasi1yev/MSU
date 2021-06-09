all:
	mpic++ -std=c++11 -O2 -Wall -Werror -fopenmp -c q_gates.cpp
	ar r libq_gates.a q_gates.o

run_test: main test
	mpirun -np 1 ./main 2 6 test_data.dat test_proc_data_01.dat
	mpirun -np 2 ./main 2 6 test_data.dat test_proc_data_02.dat
	mpirun -np 4 ./main 2 6 test_data.dat test_proc_data_03.dat
	./test test_proc_data.dat test_proc_data_01.dat
	./test test_proc_data.dat test_proc_data_02.dat
	./test test_proc_data.dat test_proc_data_03.dat

main: main.o
	mpic++ -std=c++11 -O2 -Wall -Werror -fopenmp -o main main.o -L. -lq_gates

main.o:
	mpic++ -std=c++11 -O2 -Wall -Werror -fopenmp -c main.cpp

test:
	g++ -std=c++11 -O2 -Wall -Werror test.cpp -o test

clean:
	rm *.o main test test_proc_data_0* libq*

cpplint:
	cpplint --linelength=100 --filter=-legal,-build,-read main.cpp test.cpp