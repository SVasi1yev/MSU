run_test: main
#	./test
	#mpirun -np 1 ./main 4
	mpirun -np 2 ./main 10
	#mpirun -np 3 ./main	4
	#mpirun -np 4 ./main	4

main:  Reflector.o main.o
	mpic++ -std=c++11 -O2 -o main main.o Reflector.o

main.o: main.cpp
	mpic++ -std=c++11 -O2 -c main.cpp

Reflector.o: Reflector.h Reflector.cpp
	mpic++ -std=c++11 -O2 -c Reflector.cpp

clean:
	rm *.o main