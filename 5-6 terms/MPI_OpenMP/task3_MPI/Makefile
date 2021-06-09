all: main
	rm -rf times.dat
	mpirun -np 1 ./main 1 100000000 res.dat times.dat
	mpirun -np 2 ./main 1 100000000 res.dat times.dat
	mpirun -np 4 ./main 1 100000000 res.dat times.dat
	mpirun -np 8 ./main 1 100000000 res.dat times.dat
	mpirun -np 16 ./main 1 100000000 res.dat times.dat
	mpirun -np 32 ./main 1 100000000 res.dat times.dat
	gnuplot GNUPlotScript.plt	

clean:
	rm -rf main *.dat

main: main.cpp
	mpic++ -std=c++11 -o main main.cpp
