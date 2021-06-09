all: main
	rm -rf times.dat
	./main 1 100000000 res.dat times.dat 1
	./main 1 100000000 res.dat times.dat 2
	./main 1 100000000 res.dat times.dat 4
	./main 1 100000000 res.dat times.dat 8
	./main 1 100000000 res.dat times.dat 16
	./main 1 100000000 res.dat times.dat 32
	./main 1 100000000 res.dat times.dat 64
	./main 1 100000000 res.dat times.dat 128
	gnuplot GNUPlotScript.plt	

clean:
	rm -rf main *.dat

main: main.cpp
	g++ -std=c++11 main.cpp -o main -lpthread
